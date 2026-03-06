import os
import json
import argparse
import re
import ast
from collections import defaultdict
import spacy
from tqdm import tqdm
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from lemminflect import getLemma, getInflection


lemma_noun_cache = {}
lemma_verb_cache = {}
third_verb_cache = {}
past_verb_cache = {}


def lemmatize_noun(word):
    word_lower = word.lower().strip()
    if word_lower in lemma_noun_cache:
        return lemma_noun_cache[word_lower]
    lemma = getLemma(word.lower().strip(), upos='NOUN')[0]
    lemma_noun_cache[word_lower] = lemma
    return lemma


def lemmatize_verb(word):
    word_lower = word.lower().strip()
    if word_lower in lemma_verb_cache:
        return lemma_verb_cache[word_lower]
    lemma = getLemma(word.lower().strip(), upos='VERB')[0]
    
    lemma_verb_cache[word_lower] = lemma
    return lemma

def third_verb(word):
    word_lower = word.lower().strip()
    if word_lower in third_verb_cache:
        return third_verb_cache[word_lower]
    third_verb = getInflection(word_lower, tag='VBZ')
    third_verb_cache[word_lower] = third_verb
    return third_verb

def past_verb(word):
    word_lower = word.lower().strip()
    if word_lower in past_verb_cache:
        return past_verb_cache[word_lower]
    past_verb = getInflection(word_lower, tag='VBD')
    past_verb_cache[word_lower] = past_verb
    return past_verb

def parse_grounded_caption(caption, nlp=None):
    grounded_pattern  = r'<.*?>.*?<.*?>'
    clean_caption = caption

    res = re.findall(grounded_pattern, clean_caption)
    for item in res:
        clean_item = re.sub(r'<.*?>', '', item)
        if item.startswith('<box>'):
            clean_caption = clean_caption.replace(item, '')
        else:
            clean_caption = clean_caption.replace(item, clean_item)
    # doc = nlp(clean_caption)
    
    # 提取结构化信息
    return {        
        "original_frame": caption,
        "clean_frame": clean_caption,
        "semantic_roles": extract_roles(caption),
        # "verbs": extract_spacy_verbs(doc),
        # "main_verb": find_main_verb(doc)
    }

def extract_roles(caption):
    # 提取角色的原有逻辑（同之前代码）
    roles = []
    pattern = r"<(\w+)>(.*?)</\1><box>\[(.*?)\]</box>"
    # matches = re.findall(pattern, caption)
    for match in re.findall(pattern, caption):
        role_type, noun, box = match[0], match[1], match[2]
        roles.append({
            "role": role_type,
            "noun": noun,
            "bounding_box": parse_bbox(box)
        })
    return roles

def parse_bbox(box_str):
    # if box_str == '-1,-1,-1,-1': 
    #     return None
    try:
        return list(map(int, box_str.split(',')))
    except Exception as e:
        print(e)
        return [-1, -1, -1, -1]

        
def iou(box1, box2):
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


def eval_gsr(results, formu_solus, output_path, synonyms, nlp=None):

    final_output = []
    res = defaultdict(list)

    verb_list = []
    noun_value_list = []
    noun_value_all_list = []
    bbox_value_list = []
    bbox_value_all_list = []
    total = min(len(results), len(formu_solus))
    progress_bar = tqdm(enumerate(zip(results, formu_solus)), total=total)

    for _, (item, gt_gsr) in progress_bar:
        original_output = item['response']
        gt_gsr = ast.literal_eval(gt_gsr)
        question = item['messages'][0]['content']
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        if '<think>' in question:
            content_answer_match = re.search(answer_tag_pattern, original_output, re.DOTALL | re.VERBOSE | re.IGNORECASE)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()

            else:
                content_answer = ''
        else:
            content_answer = original_output
            
        pred_gsr = parse_grounded_caption(
            caption=content_answer,
            nlp=nlp,
            # synonyms=synonyms,
        )

        # verb accuracy
        gt_verb = gt_gsr['main_verb']
        lemmatize_gt_verb = lemmatize_verb(gt_gsr['main_verb'])
        split_word = list(pred_gsr['clean_frame'].split(' '))
        verb_match = (
                lemmatize_gt_verb in split_word
                or gt_verb in split_word
                or len(set(third_verb(lemmatize_gt_verb)) & set(split_word)) > 0
                or len(set(past_verb(lemmatize_gt_verb)) & set(split_word)) > 0 
                        )
        verb_list.append(verb_match)

        # import pdb; pdb.set_trace()
        # noun value & bbox
        per_image_noun_match = []
        per_image_bbox_match = []


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
                    lemmatize_pred_role =  lemmatize_noun(pred_role)
                    lemmatize_pred_noun = lemmatize_noun(pred_noun)
   
                    noun_match = (verb_match
                                  and (pred_role.lower().strip() == gt_role.lower().strip() or lemmatize_pred_role == gt_role.lower().strip())
                                  and (pred_noun.lower().strip() in gt_noun or lemmatize_pred_noun in gt_noun)
                                  ) 
                    
                    if gt_bbox == [-1, -1, -1, -1]:
                        bbox_match = noun_match
                    else:
                        noun_iou = iou(pred_bbox, gt_bbox)

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
            noun_value_list.append(noun_match or noun_match_num > 0)

            # if gt_bbox != [-1, -1, -1, -1]:
            per_image_bbox_match.append(bbox_match)
            bbox_value_list.append(bbox_match)

        # Create a result dictionary for this example
        result = {
            'question': item['messages'][0]['content'],
            'ground_truth': gt_gsr,
            'model_output': original_output,
            'extracted_answer': pred_gsr,
        }
        final_output.append(result)


        # 处理名词匹配结果
        if per_image_noun_match:
            noun_value_all_list.append(all(per_image_noun_match))

        # 处理边界框匹配结果
        if per_image_bbox_match:
            bbox_value_all_list.append(all(per_image_bbox_match))


    verb_score = sum(verb_list) / len(verb_list)
    noun_value_score = sum(noun_value_list) / len(noun_value_list)
    noun_value_all_score = sum(noun_value_all_list) / len(noun_value_all_list)
    grnd_value_score = sum(bbox_value_list) / len(bbox_value_list)
    grnd_value_all_score = sum(bbox_value_all_list) / len(bbox_value_all_list)

    print(f'verb: {verb_score * 100:.2f}')
    print(f'value: {noun_value_score * 100:.2f}')
    print(f'value-all: {noun_value_all_score * 100:.2f}')
    print(f'grnd value: {grnd_value_score * 100:.2f}')
    print(f'grnd value-all: {grnd_value_all_score * 100:.2f}')



    # Save results to a JSON file
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        json.dump({
            'verb': verb_score,
            'value': noun_value_score,
            'value_all': noun_value_all_score,
            'value': grnd_value_score,
            'grnd_value:': grnd_value_all_score,
            'grnd_value_all': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print("-"*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--formu_solu_path", type=str)
    parser.add_argument("--synonyms_path", type=str)
    args = parser.parse_args()

    synonyms = {}
    with open(args.synonyms_path, 'r') as f:
        for line in f:
            phrases = line.strip().split(',')
            key = phrases[0].strip()
            for p in phrases[1:]:
                synonyms[p.strip()] = key


    results = [json.loads(line) for line in open(args.result_path)]
    with open(args.formu_solu_path, 'r') as f:
        formu_solus_data = json.load(f)

    formu_solus = []
    for item in formu_solus_data:
        formu_solus.append(item['solution'])
    eval_gsr(results, formu_solus, args.output_path, synonyms)