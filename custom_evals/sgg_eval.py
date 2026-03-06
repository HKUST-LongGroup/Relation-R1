import os
import json
import argparse
import re
import ast
from collections import defaultdict
import spacy
from tqdm import tqdm



def postprocess_text(text_list, synonyms, nlp):
    if isinstance(text_list, str):
        text_list = [text_list]

    synonyms_text_list = []
    for text in text_list:
        text_nlp_list = nlp(text)
        for text_nlp in text_nlp_list:
            text = text_nlp.lemma_
            synonyms_text_list.append(synonyms[text] if text in synonyms else text)
    return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

def extract_objects(
    grounded_caption: str,
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

    last_tag = REF_START_TAG
    last_tag_value = 'Unknown'
    for item in res:
        clean_item = re.sub(r'<.*?>', '', item)

        if item.startswith(BOX_START_TAG):
            clean_caption = clean_caption.replace(item, '')
            try:
                clean_item = json.loads(clean_item)
            except:
                continue
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

    return objects, relations, bbox2category

def parse_scene_graph(scene_graph_caption, synonyms, nlp):
    _, relations, bbox2category = extract_objects(scene_graph_caption)

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

    parsed_scene_graph = []

    for t in scene_graph:
        try:
            t[0] = postprocess_text(t[0], synonyms=synonyms, nlp=nlp)
            t[2] = postprocess_text(t[2], synonyms=synonyms, nlp=nlp)
    
            for subj in t[0]:
                for obj in t[2]:
                    parsed_scene_graph.append([subj, t[1], obj, t[3], t[4]])

        except Exception as e:
            print(e)
            print(f'Fail to parse {scene_graph_caption}')

    return parsed_scene_graph, correct_format

        
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


def eval_sgg(results, output_path, synonyms, nlp):

    final_output = []
  

    res = defaultdict(list)
    recall = []
    mean_recall = defaultdict(list)

    for item in tqdm(results):
        # original_output = item['response']
        original_output = item['response']
        ground_truth = ast.literal_eval(item['labels'])
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        question = item['messages'][0]['content']

        gt_scene_graph = []
        for t in ground_truth:
            gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = t

            # gold_subj = gold_subj.replace('-merged', '').replace('-other', '')
            gold_subj = gold_subj.split('-')[0]

            # gold_obj = gold_obj.replace('-merged', '').replace('-other', '')
            gold_obj = gold_obj.split('-')[0]

            gt_scene_graph.append([gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred])

        if '<think>' in question:
            content_answer_match = re.search(answer_tag_pattern, original_output, re.DOTALL | re.VERBOSE | re.IGNORECASE)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()

            else:
                content_answer = ''
        else:
            content_answer = original_output
        pred_scene_graph, correct_format = parse_scene_graph(
            scene_graph_caption=content_answer,
            synonyms=synonyms,
            nlp=nlp
        )
        # import pdb; pdb.set_trace()

        res['correct_format_ratio'].append(correct_format)
        res['num_pred_tuples'].append(len(pred_scene_graph))

        for gold in gt_scene_graph:
            match = False
            for pred in pred_scene_graph:
                try:
                    pred_subj, pred_subj_bbox, pred_obj, pred_obj_bbox, pred_pred = pred
                    gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = gold

                    subj_iou = iou(pred_subj_bbox, gold_subj_bbox)
                    obj_iou = iou(pred_obj_bbox, gold_obj_bbox)

                    match = (
                        pred_subj == gold_subj
                        and subj_iou >= 0.5
                        and pred_obj == gold_obj
                        and obj_iou >= 0.5
                        and pred_pred == gold_pred
                    )
                    if match:
                        break
                except Exception as e:
                    # import pdb; pdb.set_trace()
                    print(e)
                    continue
            recall.append(match)
            mean_recall[gold[-1]].append(match)
        # import pdb;pdb.set_trace()
        # Create a result dictionary for this example
        result = {
            'question': item['messages'][0]['content'],
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': pred_scene_graph,
        }
        final_output.append(result)



    mean_recall_list = []
    for k, v in mean_recall.items():
        mean_recall_list.append(sum(v) / len(v))
        print(f'Recall({k}): {sum(v) / len(v) * 100:.2f}')

    recall_score = sum(recall) / len(recall)
    mrecall_score = sum(mean_recall_list) / len(mean_recall_list)
    print(f'Recall: {recall_score * 100:.2f}')
    print(f'Mean Recall for {len(mean_recall_list)} predicates: {mrecall_score * 100:.2f}')


    # Save results to a JSON file
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        json.dump({
            'recall': recall_score,
            'mean_recall': mrecall_score,
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print("-"*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--synonyms_path", type=str)
    args = parser.parse_args()

    synonyms = {}
    with open(args.synonyms_path, 'r') as f:
        for line in f:
            phrases = line.strip().split(',')
            key = phrases[0].strip()
            for p in phrases[1:]:
                synonyms[p.strip()] = key

    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    results = [json.loads(line) for line in open(args.result_path)]
    eval_sgg(results, args.output_path, synonyms, nlp)

