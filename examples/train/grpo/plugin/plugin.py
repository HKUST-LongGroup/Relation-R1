import imp
import re
from typing import List

from swift.plugin import ORM, orms
from swift.utils import get_logger
import torch
import spacy
from collections import defaultdict
import numpy as np
import json


logger = get_logger()

from lemminflect import getLemma, getInflection
import ast


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards



class IoU_Rewards(ORM):
    def iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []

        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL | re.MULTILINE)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL | re.MULTILINE)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        if self.iou(bbox, sol) > 0.5:
                            reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
        return rewards
    

class SGG_Rewards(ORM):
    def __init__(self) -> None:
        super().__init__()

        synonyms_file = "custom_evals/synonyms.txt"
        synonyms = {}
        with open(synonyms_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            phrases = line.split(',')
            for p in phrases:
                synonyms[p.strip()] = phrases[0].strip()

        spacy_model = "en_core_web_sm"
        self.synonyms = synonyms
        self.nlp = spacy.load(spacy_model)

    def postprocess_text(self, text_list, synonyms):
        if isinstance(text_list, str):
            text_list = [text_list]

        synonyms_text_list = []
        for text in text_list:
            # the sky --> ['the', 'sky']
            text_nlp_list = self.nlp(text)
            for text_nlp in text_nlp_list:
                text = text_nlp.lemma_
                synonyms_text_list.append(synonyms[text] if text in synonyms else text)
        return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

    def iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    def extract_objects(self, grounded_caption: str,
    grounded_pattern: str = r'<.*?>.*?<.*?>',
    ):
        objects = defaultdict(list)
        relations = defaultdict(list)
        REF_START_TAG = '<ref>'
        REF_END_TAG = '</ref>'
        BOX_START_TAG = '<box>'
        BOX_END_TAG = '</box>'
        REL_START_TAG = '<pred>'
        REL_END_TAG = '</pred>'
        clean_caption = grounded_caption
        clean_caption = clean_caption.replace(REF_START_TAG, '').replace(REF_END_TAG, '')
        clean_caption = clean_caption.replace(REL_START_TAG, '').replace(REL_END_TAG, '')
        res = re.findall(grounded_pattern, grounded_caption)
        last_tag = None
        last_tag_value = None
        for item in res:
            clean_item = re.sub(r'<.*?>', '', item)
            if item.startswith(BOX_START_TAG):
                clean_caption = clean_caption.replace(item, '')
                try:
                    clean_item = json.loads(clean_item)
                except Exception as e:
                    print('Invalid format:', clean_item)
                    raise e
                if last_tag == REF_START_TAG:
                    objects[last_tag_value].extend(clean_item)
                elif last_tag == REL_START_TAG:
                    relations[last_tag_value].append(clean_item)
                else:
                    raise NotImplementedError(grounded_caption)
            else:
                last_tag = REF_START_TAG if item.startswith(REF_START_TAG) else REL_START_TAG
                last_tag_value = clean_item

        bbox2category = defaultdict(list)
        for k, v in objects.items():
            for bbox in v:
                bbox2category[json.dumps(bbox)].append(k)

        return objects, relations, bbox2category, clean_caption
    
    def parse_scene_graph(self, scene_graph_caption):
        _, relations, bbox2category, clean_caption = self.extract_objects(scene_graph_caption)

        scene_graph = []
        correct_format = True
        for rel_name, bbox_list in relations.items():
            if len(bbox_list) % 2 != 0:
                correct_format = False

            for i in range(0, len(bbox_list), 2):
                if i+1 >= len(bbox_list):
                    continue

                subject_bboxes = bbox_list[i]
                object_bboxes = bbox_list[i+1]

                if len(subject_bboxes) == 1:
                    subject_bboxes = subject_bboxes * len(object_bboxes)

                if len(object_bboxes) == 1:
                    object_bboxes = object_bboxes * len(subject_bboxes)

                if len(subject_bboxes) != len(object_bboxes):
                    correct_format = False

                for subj_bbox, obj_bbox in zip(subject_bboxes, object_bboxes):
                    subj = bbox2category[json.dumps(subj_bbox)]
                    obj = bbox2category[json.dumps(obj_bbox)]
                    scene_graph.append([subj, subj_bbox, obj, obj_bbox, rel_name])

        parsed_sgg = []
        for t in scene_graph:
            try:
                t[0] = self.postprocess_text(t[0], synonyms=self.synonyms)
                t[2] = self.postprocess_text(t[2], synonyms=self.synonyms)

                for subj in t[0]:
                    for obj in t[2]:
                        parsed_sgg.append([subj, t[1], obj, t[3], t[4]])
            except Exception as e:
                    print(e)
                    print(f'Fail to parse {scene_graph_caption}')                
        return parsed_sgg  

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []

        answer_tag_pattern = r'<answer>(.*?)</answer>'
        rewards = []
        for content, sol in zip(completions, solution):
            if '<ref>' not in sol:
                rewards.append(0.0)
                continue
            reward = 0.0
            # Try symbolic verification first

            # if torch.cuda.current_device() == 0:
            #     import pdb;pdb.set_trace()

            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    recall = []
                    mean_recall = defaultdict(list)

                    pred_sgg = self.parse_scene_graph(content_answer)
                    gt_sgg = self.parse_scene_graph(sol)
                    for gold in gt_sgg:
                        match = False
                        for pred in pred_sgg:
                            pred_subj, pred_subj_bbox, pred_obj, pred_obj_bbox, pred_pred = pred
                            gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = gold

                            subj_iou = self.iou(pred_subj_bbox, gold_subj_bbox)
                            obj_iou = self.iou(pred_obj_bbox, gold_obj_bbox)

                            match = (
                                pred_subj == gold_subj
                                and subj_iou >= 0.5
                                and pred_obj == gold_obj
                                and obj_iou >= 0.5
                                and pred_pred == gold_pred
                            )
                            if match:
                                break
                        recall.append(match)
                        mean_recall[gold[-1]].append(match)
                    mean_recall_list = []
                    for k, v in mean_recall.items():
                        mean_recall_list.append(sum(v) / len(v))
                        # print(f'Recall({k}): {sum(v) / len(v) * 100:.2f}')

                    recall_score = sum(recall) / len(recall)
                    mean_reall_score = sum(mean_recall_list) / len(mean_recall_list)
                    reward = recall_score + mean_reall_score 

            except Exception as e:
                print(e)
                pass  # Continue to next verification method if this fails
                
            rewards.append(reward)

        return rewards


class GSR_Rewards(ORM):
    def __init__(self) -> None:
        super().__init__()

        synonyms_file = "custom_evals/synonyms.txt"
        synonyms = {}
        with open(synonyms_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            phrases = line.split(',')
            for p in phrases:
                synonyms[p.strip()] = phrases[0].strip()

        self.synonyms = synonyms
        self.lemma_noun_cache = {}
        self.lemma_verb_cache = {}
        self.third_verb_cache = {}
        self.past_verb_cache = {}



    def postprocess_text(self, text_list, synonyms):
        if isinstance(text_list, str):
            text_list = [text_list]

        synonyms_text_list = []
        for text in text_list:
            text_nlp_list = self.nlp(text)
            for text_nlp in text_nlp_list:
                text = text_nlp.lemma_
                synonyms_text_list.append(synonyms[text] if text in synonyms else text)
        return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

    def iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
 
    
    def lemmatize_noun(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.lemma_noun_cache:
            return self.lemma_noun_cache[word_lower]
        lemma = getLemma(word.lower().strip(), upos='NOUN')[0]
        self.lemma_noun_cache[word_lower] = lemma
        return lemma


    def lemmatize_verb(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.lemma_verb_cache:
            return self.lemma_verb_cache[word_lower]
        # lemma = verb_lemmatizer.lemmatize(word.lower().strip(), pos='v')
        lemma = getLemma(word.lower().strip(), upos='VERB')[0]
        
        self.lemma_verb_cache[word_lower] = lemma
        return lemma

    def third_verb(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.third_verb_cache:
            return self.third_verb_cache[word_lower]
        third_verb = getInflection(word_lower, tag='VBZ')
        self.third_verb_cache[word_lower] = third_verb
        return third_verb

    def past_verb(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.past_verb_cache:
            return self.past_verb_cache[word_lower]
        past_verb = getInflection(word_lower, tag='VBD')
        self.past_verb_cache[word_lower] = past_verb
        return past_verb


    def extract_roles(self, caption):
        roles = []
        pattern = r"<(\w+)>(.*?)</\1><box>\[(.*?)\]</box>"
        # matches = re.findall(pattern, caption)
        for match in re.findall(pattern, caption):
            role_type, noun, box = match[0], match[1], match[2]
            roles.append({
                "role": role_type,
                "noun": noun,
                "bounding_box": self.parse_bbox(box)
            })
        return roles

    def parse_bbox(self, box_str):
        # if box_str == '-1,-1,-1,-1': 
        #     return None
        try:
            return list(map(int, box_str.split(',')))
        except Exception as e:
            print(e)
            return [-1, -1, -1, -1]
        

    def parse_grounded_caption(self, caption):
        grounded_pattern  = r'<.*?>.*?<.*?>'
        clean_caption = caption

        res = re.findall(grounded_pattern, clean_caption)
        for item in res:
            clean_item = re.sub(r'<.*?>', '', item)
            if item.startswith('<box>'):
                clean_caption = clean_caption.replace(item, '')
            else:
                clean_caption = clean_caption.replace(item, clean_item)

        return {        
            "original_frame": caption,
            "clean_frame": clean_caption,
            "semantic_roles": self.extract_roles(caption),
        }

 

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []

        answer_tag_pattern = r'<answer>(.*?)</answer>'
        rewards = []
        correct = []
        for content, solu in zip(completions, solution):
            if '<ref>' in solu:
                rewards.append(0.0)
                continue
            reward = 0.0

            # if torch.cuda.current_device() == 0:
            #     import pdb;pdb.set_trace()

            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    pred_gsr = self.parse_grounded_caption(content_answer)
                    gt_gsr = ast.literal_eval(solu)

                    # if torch.cuda.current_device() == 0:
                    #     import pdb;pdb.set_trace()
                    # verb correct
                    gt_verb = gt_gsr['main_verb']
                    lemmatize_gt_verb = self.lemmatize_verb(gt_gsr['main_verb'])
                    split_word = list(pred_gsr['clean_frame'].split(' '))
                    verb_match = (
                            lemmatize_gt_verb in split_word
                            or gt_verb in split_word
                            or len(set(self.third_verb(lemmatize_gt_verb)) & set(split_word)) > 0
                            or len(set(self.past_verb(lemmatize_gt_verb)) & set(split_word)) > 0 
                                    )
                    per_image_noun_match = []
                    per_image_bbox_match = []

                    # noun value & bbox
                    for gt in gt_gsr["semantic_roles"]:
                        bbox_match = False
                        noun_match = False
                        noun_match_num = 0
                        gt_role, gt_noun, gt_bbox = gt['role'], gt['noun'], gt['bounding_box']
                        for pred in pred_gsr["semantic_roles"]:
                            pred_role, pred_noun, pred_bbox = pred['role'], pred['noun'], pred['bounding_box']

                            # print(torch.tensor(pred_subj_bbox).shape)
                            try:
                                # noun_value:
                                lemmatize_pred_role =  self.lemmatize_noun(pred_role)
                                lemmatize_pred_noun = self.lemmatize_noun(pred_noun)
            
                                noun_match = (verb_match
                                            and (pred_role.lower().strip() == gt_role.lower().strip() or lemmatize_pred_role == gt_role.lower().strip())
                                            and (pred_noun.lower().strip() in gt_noun or lemmatize_pred_noun in gt_noun)
                                            ) 
                                if gt_bbox == [-1, -1, -1, -1]:
                                    bbox_match = noun_match
                                else:
                                    noun_iou = self.iou(pred_bbox, gt_bbox)

                                    bbox_match = (
                                        noun_match
                                        and noun_iou >= 0.5
                                    )
                            except Exception as e:
                                continue

                            if noun_match:
                                noun_match_num += 1
                            if noun_match and bbox_match:
                                break

                        per_image_noun_match.append(noun_match or noun_match_num > 0)

                        if gt_bbox != [-1, -1, -1, -1]:
                            per_image_bbox_match.append(bbox_match)


                    noun_accuracy = sum(per_image_noun_match) / (len(per_image_noun_match) + 1e-7)
                    box_accuracy = sum(per_image_bbox_match) / (len(per_image_bbox_match) + 1e-7)
                    reward = noun_accuracy + box_accuracy
            except Exception as e:
                print(e)
                pass  # Continue to next verification method if this fails

            rewards.append(reward)

        return rewards

class Obj_Rewards(ORM):

    def postprocess_text(self, text_list, synonyms):
        if isinstance(text_list, str):
            text_list = [text_list]

        synonyms_text_list = []
        for text in text_list:
            # the sky --> ['the', 'sky']
            text_nlp_list = self.nlp(text)
            for text_nlp in text_nlp_list:
                text = text_nlp.lemma_
                synonyms_text_list.append(synonyms[text] if text in synonyms else text)
        return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

    def iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    def extract_objects(self, grounded_caption: str,
    grounded_pattern: str = r'<.*?>.*?<.*?>',
    ):
        objects = defaultdict(list)
        relations = defaultdict(list)
        REF_START_TAG = '<ref>'
        REF_END_TAG = '</ref>'
        BOX_START_TAG = '<box>'
        BOX_END_TAG = '</box>'
        REL_START_TAG = '<pred>'
        REL_END_TAG = '</pred>'
        clean_caption = grounded_caption
        clean_caption = clean_caption.replace(REF_START_TAG, '').replace(REF_END_TAG, '')
        clean_caption = clean_caption.replace(REL_START_TAG, '').replace(REL_END_TAG, '')
        res = re.findall(grounded_pattern, grounded_caption)
        last_tag = None
        last_tag_value = None
        for item in res:
            clean_item = re.sub(r'<.*?>', '', item)
            if item.startswith(BOX_START_TAG):
                clean_caption = clean_caption.replace(item, '')
                try:
                    clean_item = json.loads(clean_item)
                except Exception as e:
                    print('Invalid format:', clean_item)
                    raise e
                if last_tag == REF_START_TAG:
                    objects[last_tag_value].extend(clean_item)
                elif last_tag == REL_START_TAG:
                    relations[last_tag_value].append(clean_item)
                else:
                    raise NotImplementedError(grounded_caption)
            else:
                last_tag = REF_START_TAG if item.startswith(REF_START_TAG) else REL_START_TAG
                last_tag_value = clean_item

        bbox2category = defaultdict(list)
        for k, v in objects.items():
            for bbox in v:
                bbox2category[json.dumps(bbox)].append(k)

        return objects, relations, bbox2category, clean_caption
    

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []

        answer_tag_pattern = r'<answer>(.*?)</answer>'
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first

            # if torch.cuda.current_device() == 0:
            #     import pdb;pdb.set_trace()
            try:
                pred_answer = re.search(answer_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                pred_answer = pred_answer.group(1).strip() if pred_answer else ""
                gt_answer = sol

                _, _, pred_bbox2cat, _ = self.extract_objects(pred_answer)
                _, _, gt_bbox2cat, _ = self.extract_objects(gt_answer)

                total_boxes = len(gt_bbox2cat) 
                matched_boxes = 0

                for gt_bbox_str, gt_cats in gt_bbox2cat.items():
                    gt_box = json.loads(gt_bbox_str)
                    matched = False
                    
                    for pred_bbox_str, pred_cats in pred_bbox2cat.items():
                        
                        pred_box = json.loads(pred_bbox_str)
                        
                        if self.iou(gt_box, pred_box) < 0.5:
                            continue
                        
                        if not set(gt_cats).intersection(pred_cats):
                            continue

                        matched_boxes += 1
                        matched = True
                        break
                    
                    if matched:
                        del pred_bbox2cat[pred_bbox_str]

                reward = (matched_boxes / total_boxes if total_boxes > 0 else 0.0) * 2
                rewards.append(reward)

            except Exception as e:
                print(f"Object reward error: {str(e)}")
                rewards.append(0.0)
        return rewards  


class Merge_Rewards(ORM):
    def __init__(self) -> None:
        super().__init__()

        self.sgg_rewards = SGG_Rewards()
        self.gsr_rewards = GSR_Rewards()


    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = [0.0 for i in range(len(solution))]

        # judge task
        sgg_ids = [i for i, s in enumerate(solution) if '<ref>' in s]
        gsr_ids = [i for i, s in enumerate(solution) if '<ref>' not in s]
        # SGG

        if sgg_ids:
            sgg_subset = [completions[i] for i in sgg_ids]
            sgg_solutions = [solution[i] for i in sgg_ids]
            sgg_rewards = self.sgg_rewards(sgg_subset, sgg_solutions)
            for idx, reward in zip(sgg_ids, sgg_rewards):
                rewards[idx] = reward
        # GSR
        if gsr_ids:

            gsr_completions = [completions[i] for i in gsr_ids]
            gsr_solutions = [solution[i] for i in gsr_ids]
            gsr_rewards = self.gsr_rewards(gsr_completions, gsr_solutions)
        
            for idx, reward in zip(gsr_ids, gsr_rewards):
                rewards[idx] = reward

        # if torch.cuda.current_device() == 0:
        #     import pdb;pdb.set_trace()

        return rewards


class GSR_CoT_Rewards(ORM):
    def __init__(self) -> None:
        super().__init__()

        synonyms_file = "custom_evals/synonyms.txt"
        synonyms = {}
        with open(synonyms_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            phrases = line.split(',')
            for p in phrases:
                synonyms[p.strip()] = phrases[0].strip()

        self.synonyms = synonyms
        self.lemma_noun_cache = {}
        self.lemma_verb_cache = {}
        self.third_verb_cache = {}
        self.past_verb_cache = {}


    # TO DO
    def postprocess_text(self, text_list, synonyms):
        if isinstance(text_list, str):
            text_list = [text_list]

        synonyms_text_list = []
        for text in text_list:
            # the sky --> ['the', 'sky']
            text_nlp_list = self.nlp(text)
            for text_nlp in text_nlp_list:
                text = text_nlp.lemma_
                synonyms_text_list.append(synonyms[text] if text in synonyms else text)
        return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

    def iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
 
    
    def lemmatize_noun(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.lemma_noun_cache:
            return self.lemma_noun_cache[word_lower]
        lemma = getLemma(word.lower().strip(), upos='NOUN')[0]
        self.lemma_noun_cache[word_lower] = lemma
        return lemma


    def lemmatize_verb(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.lemma_verb_cache:
            return self.lemma_verb_cache[word_lower]
        # lemma = verb_lemmatizer.lemmatize(word.lower().strip(), pos='v')
        lemma = getLemma(word.lower().strip(), upos='VERB')[0]
        
        self.lemma_verb_cache[word_lower] = lemma
        return lemma

    def third_verb(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.third_verb_cache:
            return self.third_verb_cache[word_lower]
        third_verb = getInflection(word_lower, tag='VBZ')
        self.third_verb_cache[word_lower] = third_verb
        return third_verb

    def past_verb(self, word):
        word_lower = word.lower().strip()
        if word_lower in self.past_verb_cache:
            return self.past_verb_cache[word_lower]
        past_verb = getInflection(word_lower, tag='VBD')
        self.past_verb_cache[word_lower] = past_verb
        return past_verb


    def extract_roles(self, caption):
        # 提取角色的原有逻辑（同之前代码）
        roles = []
        pattern = r"<(\w+)>(.*?)</\1><box>\[(.*?)\]</box>"
        # matches = re.findall(pattern, caption)
        for match in re.findall(pattern, caption):
            role_type, noun, box = match[0], match[1], match[2]
            roles.append({
                "role": role_type,
                "noun": noun,
                "bounding_box": self.parse_bbox(box)
            })
        return roles

    def parse_bbox(self, box_str):
        # if box_str == '-1,-1,-1,-1': 
        #     return None
        try:
            return list(map(int, box_str.split(',')))
        except Exception as e:
            print(e)
            return [-1, -1, -1, -1]
        

    def parse_grounded_caption(self, caption):
        grounded_pattern  = r'<.*?>.*?<.*?>'
        clean_caption = caption

        res = re.findall(grounded_pattern, clean_caption)
        for item in res:
            clean_item = re.sub(r'<.*?>', '', item)
            if item.startswith('<box>'):
                clean_caption = clean_caption.replace(item, '')
        # spaCy语法分析
            else:
                clean_caption = clean_caption.replace(item, clean_item)

        # 提取结构化信息
        return {        
            "original_frame": caption,
            "clean_frame": clean_caption,
            "semantic_roles": self.extract_roles(caption),
        }

 

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []

        cot_tag_pattern = r'<think>(.*?)</think>'
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        rewards = []

        for content, solu in zip(completions, solution):
            reward = 0.0

            # if torch.cuda.current_device() == 0:
            #     import pdb;pdb.set_trace()

            try:
                gt_gsr = ast.literal_eval(solu)
                content_cot_match = re.search(cot_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                if content_cot_match:
                    content_cot = content_cot_match.group(1).strip()

                    # Judge Activity
                    all_type_verbs = []
                    gt_verb = gt_gsr['main_verb']
                    lemmatize_gt_verb = self.lemmatize_verb(gt_gsr['main_verb'])
                    all_type_verbs = [gt_verb, lemmatize_gt_verb]
                    all_type_verbs.extend(self.third_verb(lemmatize_gt_verb))
                    all_type_verbs.extend(self.past_verb(lemmatize_gt_verb))

                    verb_match = False
                    for verb in all_type_verbs:
                        if verb in content_cot:
                            verb_match = True
                            break

                    role_match_list = []
                    
                    noun_match_list = []

                    box_match_list = []

                    box_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
                    cot_bboxes = re.findall(box_pattern, content_cot)
                    cot_bboxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in cot_bboxes]
                    # print(cot_bboxes)

                    for gt in gt_gsr["semantic_roles"]:
                        # Judge Role 
                        gt_role, gt_noun, gt_bbox = gt['role'], gt['noun'], gt['bounding_box']
                        role_match =  gt_role in content_cot.lower()
                        role_match_list.append(role_match)

                        # Judge Noun 
                        noun_match = False
                        for gt_noun_item in gt_noun:
                            if gt_noun_item in content_cot.lower():
                                noun_match = True
                                break
                        noun_match_list.append(noun_match)

                        # Judge Box
                        if gt_bbox != [-1, -1, -1, -1]:
                            box_match = False
                            for cot_bbox in cot_bboxes:
                                if self.iou(gt_bbox, cot_bbox) > 0.5:
                                    box_match = True
                                    break  
                            box_match_list.append(box_match)          
                        
                    
                    correctness = verb_match
                    judge_num = 1
                    for judge_list in [role_match_list, noun_match_list, box_match_list]:
                        if len(judge_list) > 0:
                            correctness += sum(judge_list)/len(judge_list)
                            judge_num += 1
                    reward += correctness/judge_num 

                    # print(verb_match, role_match_list, noun_match_list, box_match_list)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()

                    # Judge Answer Consistency

                    pred_gsr = self.parse_grounded_caption(content_answer)
                    consistency = True
                    for pred in pred_gsr["semantic_roles"]:
                        pred_role, pred_noun, pred_bbox = pred['role'], pred['noun'], pred['bounding_box']
                        role_match = pred_role in content_cot.lower()
                        noun_match = pred_noun in content_cot.lower() if pred_noun != 'null' else True
                        box_match = str(pred_bbox) in content_cot if pred_bbox != [-1, -1, -1, -1] else True
                        consistency = role_match and noun_match and box_match
                        if not consistency:
                            break
                    reward += consistency

            except Exception as e:
                print(e)
                pass  # Continue to next verification method if this fails

            rewards.append(reward)

        return rewards


class SGG_CoT_Rewards(ORM):
    def __init__(self) -> None:
        super().__init__()

        synonyms_file = "custom_evals/synonyms.txt"
        synonyms = {}
        with open(synonyms_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            phrases = line.split(',')
            for p in phrases:
                synonyms[p.strip()] = phrases[0].strip()

        spacy_model = "en_core_web_sm"
        self.synonyms = synonyms
        self.nlp = spacy.load(spacy_model)

    def postprocess_text(self, text_list, synonyms):
        if isinstance(text_list, str):
            text_list = [text_list]

        synonyms_text_list = []
        for text in text_list:
            # the sky --> ['the', 'sky']
            text_nlp_list = self.nlp(text)
            for text_nlp in text_nlp_list:
                text = text_nlp.lemma_
                synonyms_text_list.append(synonyms[text] if text in synonyms else text)
        return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

    def iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    def extract_objects(self, grounded_caption: str,
    grounded_pattern: str = r'<.*?>.*?<.*?>',
    ):
        objects = defaultdict(list)
        relations = defaultdict(list)
        REF_START_TAG = '<ref>'
        REF_END_TAG = '</ref>'
        BOX_START_TAG = '<box>'
        BOX_END_TAG = '</box>'
        REL_START_TAG = '<pred>'
        REL_END_TAG = '</pred>'
        clean_caption = grounded_caption
        clean_caption = clean_caption.replace(REF_START_TAG, '').replace(REF_END_TAG, '')
        clean_caption = clean_caption.replace(REL_START_TAG, '').replace(REL_END_TAG, '')
        res = re.findall(grounded_pattern, grounded_caption)
        last_tag = None
        last_tag_value = None
        for item in res:
            clean_item = re.sub(r'<.*?>', '', item)
            if item.startswith(BOX_START_TAG):
                clean_caption = clean_caption.replace(item, '')
                try:
                    clean_item = json.loads(clean_item)
                except Exception as e:
                    print('Invalid format:', clean_item)
                    raise e
                if last_tag == REF_START_TAG:
                    objects[last_tag_value].extend(clean_item)
                elif last_tag == REL_START_TAG:
                    relations[last_tag_value].append(clean_item)
                else:
                    raise NotImplementedError(grounded_caption)
            else:
                last_tag = REF_START_TAG if item.startswith(REF_START_TAG) else REL_START_TAG
                last_tag_value = clean_item

        bbox2category = defaultdict(list)
        for k, v in objects.items():
            for bbox in v:
                bbox2category[json.dumps(bbox)].append(k)

        return objects, relations, bbox2category, clean_caption
    
    def parse_scene_graph(self, scene_graph_caption):
        _, relations, bbox2category, clean_caption = self.extract_objects(scene_graph_caption)

        scene_graph = []
        correct_format = True
        for rel_name, bbox_list in relations.items():
            if len(bbox_list) % 2 != 0:
                correct_format = False

            for i in range(0, len(bbox_list), 2):
                if i+1 >= len(bbox_list):
                    continue

                subject_bboxes = bbox_list[i]
                object_bboxes = bbox_list[i+1]

                if len(subject_bboxes) == 1:
                    subject_bboxes = subject_bboxes * len(object_bboxes)

                if len(object_bboxes) == 1:
                    object_bboxes = object_bboxes * len(subject_bboxes)

                if len(subject_bboxes) != len(object_bboxes):
                    correct_format = False

                for subj_bbox, obj_bbox in zip(subject_bboxes, object_bboxes):
                    subj = bbox2category[json.dumps(subj_bbox)]
                    obj = bbox2category[json.dumps(obj_bbox)]
                    scene_graph.append([subj, subj_bbox, obj, obj_bbox, rel_name])

        parsed_sgg = []
        for t in scene_graph:
            try:
                # t[0] = self.postprocess_text(t[0], synonyms=self.synonyms)
                # t[2] = self.postprocess_text(t[2], synonyms=self.synonyms)

                for subj in t[0]:
                    for obj in t[2]:
                        parsed_sgg.append([subj, t[1], obj, t[3], t[4]])
            except Exception as e:
                    print(e)
                    print(f'Fail to parse {scene_graph_caption}')                
        return parsed_sgg  

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []

        cot_tag_pattern = r'<think>(.*?)</think>'
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first

            # if torch.cuda.current_device() == 0:
            #     import pdb;pdb.set_trace()

            try:
                content_cot_match = re.search(cot_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                gt_sgg = self.parse_scene_graph(sol)

                
                objs_match_list = []
                
                box_match_list = []

                rels_match_list = []
                
                if content_cot_match:
                    content_cot = content_cot_match.group(1).strip()

                    box_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
                    cot_bboxes = re.findall(box_pattern, content_cot)

                    cot_bboxes = [[float(box[0]), float(box[1]), float(box[2]), float(box[3])] for box in cot_bboxes]

                    # print(gt_sgg)
                    for gold in gt_sgg:
                        gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = gold
                        # Judge Objects
                        subj_match = gold_subj in content_cot.lower()
                        obj_match = gold_obj in content_cot.lower()
                        # if not subj_match:
                        #     print(gold_subj)
                        # if not obj_match:
                        #     print(gold_obj)

                        objs_match_list.append(subj_match)
                        objs_match_list.append(obj_match)
                        # Judge Box

                        box_match = False
                        for cot_bbox in cot_bboxes:
                            if self.iou(gold_subj_bbox, cot_bbox) > 0.5:
                                box_match = True
                                break  
                        box_match_list.append(box_match)    

                        box_match = False
                        for cot_bbox in cot_bboxes:
                            if self.iou(gold_obj_bbox, cot_bbox) > 0.5:
                                box_match = True
                                break  
                        box_match_list.append(box_match)  

                        # Judge Relations
                        rel_match =  gold_pred in content_cot.lower()
                        rels_match_list.append(rel_match)

                    correctness = 0.0
                    judge_num = 0
                    # print(objs_match_list, box_match_list, rels_match_list)
                    for judge_list in [objs_match_list, box_match_list, rels_match_list]:
                        if len(judge_list) > 0:
                            correctness += sum(judge_list)/len(judge_list)
                            judge_num += 1
                            
                    reward += correctness/(judge_num + 1e-7)

                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()

                    pred_sgg = self.parse_scene_graph(content_answer)

                    consistency = True
                    for pred in pred_sgg:
                        pred_subj, pred_subj_bbox, pred_obj, pred_obj_bbox, pred_pred = pred
                        if pred_subj not in content_cot.lower() \
                            or pred_obj not in content_cot.lower() \
                            or str(pred_subj_bbox) not in content_cot.lower() \
                            or str(pred_obj_bbox) not in content_cot.lower() \
                            or str(pred_pred) not in content_cot.lower():

                            consistency = False
                            break

                    reward += consistency

            except Exception as e:
                print(e)
                pass  # Continue to next verification method if this fails
                
            rewards.append(reward)

        return rewards


class Merge_CoT_Rewards(ORM):
    def __init__(self) -> None:
        super().__init__()

        self.sgg_rewards = SGG_CoT_Rewards()
        self.gsr_rewards = GSR_CoT_Rewards()


    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = [0.0 for i in range(len(solution))]

        # judge task
        sgg_ids = [i for i, s in enumerate(solution) if '<ref>' in s]
        gsr_ids = [i for i, s in enumerate(solution) if '<ref>' not in s]
        # SGG

        if sgg_ids:
            sgg_subset = [completions[i] for i in sgg_ids]
            sgg_solutions = [solution[i] for i in sgg_ids]
            sgg_rewards = self.sgg_rewards(sgg_subset, sgg_solutions)
            for idx, reward in zip(sgg_ids, sgg_rewards):
                rewards[idx] = reward
        # GSR
        if gsr_ids:

            gsr_completions = [completions[i] for i in gsr_ids]
            gsr_solutions = [solution[i] for i in gsr_ids]
            gsr_rewards = self.gsr_rewards(gsr_completions, gsr_solutions)
        
            for idx, reward in zip(gsr_ids, gsr_rewards):
                rewards[idx] = reward

        # if torch.cuda.current_device() == 0:
        #     import pdb;pdb.set_trace()

        return rewards


class SGGFormat_Rewards(ORM):
    def __init__(self) -> None:
        super().__init__()

        synonyms_file = "custom_evals/synonyms.txt"
        synonyms = {}
        with open(synonyms_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            phrases = line.split(',')
            for p in phrases:
                synonyms[p.strip()] = phrases[0].strip()

        spacy_model = "en_core_web_sm"
        self.synonyms = synonyms
        self.nlp = spacy.load(spacy_model)

    def postprocess_text(self, text_list, synonyms):
        if isinstance(text_list, str):
            text_list = [text_list]

        synonyms_text_list = []
        for text in text_list:
            # the sky --> ['the', 'sky']
            text_nlp_list = self.nlp(text)
            for text_nlp in text_nlp_list:
                text = text_nlp.lemma_
                synonyms_text_list.append(synonyms[text] if text in synonyms else text)
        return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

    def iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    def extract_objects(self, grounded_caption: str,
    grounded_pattern: str = r'<.*?>.*?<.*?>',
    ):
        objects = defaultdict(list)
        relations = defaultdict(list)
        REF_START_TAG = '<ref>'
        REF_END_TAG = '</ref>'
        BOX_START_TAG = '<box>'
        BOX_END_TAG = '</box>'
        REL_START_TAG = '<pred>'
        REL_END_TAG = '</pred>'
        clean_caption = grounded_caption
        clean_caption = clean_caption.replace(REF_START_TAG, '').replace(REF_END_TAG, '')
        clean_caption = clean_caption.replace(REL_START_TAG, '').replace(REL_END_TAG, '')
        res = re.findall(grounded_pattern, grounded_caption)
        last_tag = None
        last_tag_value = None
        for item in res:
            clean_item = re.sub(r'<.*?>', '', item)
            if item.startswith(BOX_START_TAG):
                clean_caption = clean_caption.replace(item, '')
                try:
                    clean_item = json.loads(clean_item)
                except Exception as e:
                    print('Invalid format:', clean_item)
                    raise e
                if last_tag == REF_START_TAG:
                    objects[last_tag_value].extend(clean_item)
                elif last_tag == REL_START_TAG:
                    relations[last_tag_value].append(clean_item)
                else:
                    raise NotImplementedError(grounded_caption)
            else:
                last_tag = REF_START_TAG if item.startswith(REF_START_TAG) else REL_START_TAG
                last_tag_value = clean_item

        bbox2category = defaultdict(list)
        for k, v in objects.items():
            for bbox in v:
                bbox2category[json.dumps(bbox)].append(k)

        return objects, relations, bbox2category, clean_caption
    
    def parse_scene_graph(self, scene_graph_caption):
        _, relations, bbox2category, clean_caption = self.extract_objects(scene_graph_caption)

        scene_graph = []
        correct_format = True
        for rel_name, bbox_list in relations.items():
            if len(bbox_list) % 2 != 0:
                correct_format = False

            for i in range(0, len(bbox_list), 2):
                if i+1 >= len(bbox_list):
                    continue

                subject_bboxes = bbox_list[i]
                object_bboxes = bbox_list[i+1]

                if len(subject_bboxes) == 1:
                    subject_bboxes = subject_bboxes * len(object_bboxes)

                if len(object_bboxes) == 1:
                    object_bboxes = object_bboxes * len(subject_bboxes)

                if len(subject_bboxes) != len(object_bboxes):
                    correct_format = False

                for subj_bbox, obj_bbox in zip(subject_bboxes, object_bboxes):
                    subj = bbox2category[json.dumps(subj_bbox)]
                    obj = bbox2category[json.dumps(obj_bbox)]
                    scene_graph.append([subj, subj_bbox, obj, obj_bbox, rel_name])

        parsed_sgg = []
        for t in scene_graph:
            try:
                # psg use synonyms
                # t[0] = self.postprocess_text(t[0], synonyms=self.synonyms)
                # t[2] = self.postprocess_text(t[2], synonyms=self.synonyms)
                # vg 
                for subj in t[0]:
                    for obj in t[2]:
                        parsed_sgg.append([subj, t[1], obj, t[3], t[4]])
            except Exception as e:
                    print(e)
                    print(f'Fail to parse {scene_graph_caption}')                
        return parsed_sgg  

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []

        answer_tag_pattern = r'<answer>(.*?)</answer>'
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first

            # if torch.cuda.current_device() == 0:
            #     import pdb;pdb.set_trace()

            try:
                ground_truth = ast.literal_eval(sol.strip())
            except Exception as e:
                    # import pdb; pdb.set_trace()
                print(e)
                ground_truth = []

            gt_scene_graph = []
            for t in ground_truth:
                gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = t

                # gold_subj = gold_subj.replace('-merged', '').replace('-other', '')
                gold_subj = gold_subj.split('-')[0]

                # gold_obj = gold_obj.replace('-merged', '').replace('-other', '')
                gold_obj = gold_obj.split('-')[0]

                gt_scene_graph.append([gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred])


            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL | re.VERBOSE | re.IGNORECASE)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    recall = []
                    mean_recall = defaultdict(list)

                try:
                    pred = ast.literal_eval(content_answer.strip())
                except Exception as e:
                    # import pdb; pdb.set_trace()
                    print(e)
                   
                pred_scene_graph = []
                for t in pred:
                    try:
                        gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = t

                        # gold_subj = gold_subj.replace('-merged', '').replace('-other', '')
                        gold_subj = gold_subj.split('-')[0]

                        # gold_obj = gold_obj.replace('-merged', '').replace('-other', '')
                        gold_obj = gold_obj.split('-')[0]

                    except Exception as e:
                        # import pdb; pdb.set_trace()
                        print(e, t)
                        continue

                    pred_scene_graph.append([gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred])


                    for gold in gt_scene_graph:
                        match = False
                        for pred in pred_scene_graph:
                            pred_subj, pred_subj_bbox, pred_obj, pred_obj_bbox, pred_pred = pred
                            gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = gold

                            subj_iou = self.iou(pred_subj_bbox, gold_subj_bbox)
                            obj_iou = self.iou(pred_obj_bbox, gold_obj_bbox)

                            match = (
                                pred_subj == gold_subj
                                and subj_iou >= 0.5
                                and pred_obj == gold_obj
                                and obj_iou >= 0.5
                                and pred_pred == gold_pred
                            )
                            if match:
                                break
                        recall.append(match)
                        mean_recall[gold[-1]].append(match)
                    mean_recall_list = []
                    for k, v in mean_recall.items():
                        mean_recall_list.append(sum(v) / len(v))
                        # print(f'Recall({k}): {sum(v) / len(v) * 100:.2f}')

                    recall_score = sum(recall) / len(recall)
                    mean_reall_score = sum(mean_recall_list) / len(mean_recall_list)
                    reward = recall_score + mean_reall_score 

            except Exception as e:
                print(e)
                pass  # Continue to next verification method if this fails
                
            rewards.append(reward)
            # return recall + mean_recall
        return rewards


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_iou'] = IoU_Rewards
orms['external_sgg'] = SGG_Rewards
orms['external_obj'] = Obj_Rewards
orms['external_gsr'] = GSR_Rewards
orms['external_gsr_cot'] = GSR_CoT_Rewards
orms['external_sgg_cot'] = SGG_CoT_Rewards
orms['external_sggformat'] = SGGFormat_Rewards
orms['external_merge'] = Merge_Rewards
orms['external_merge_cot'] = Merge_CoT_Rewards