import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from pycocotools.coco import COCO
from cocoeval import COCOEvalCap

from chair import CHAIR, load_generated_captions


def eval_chair(answer_file):
    """
    answers: json file: list[dict], keys: "caption", "image_id"
    """
    # TODO: modify the data path if you need
    anno_file = "captions_val2014.json"
    anno_dir = "/data/coco/annotations"
    
    answers = []
    for line in open(answer_file, 'r'):
        answer = json.loads(line)
        answer['caption'] = answer['text']
        answer['image_id'] = answer['question_id']
        answers.append(answer)
    
    coco = COCO(anno_file)
    formulated_output_dict = {}
    all_overall_scores = defaultdict(list)
    img_to_eval_dict = {}
    chunk_size = 100

    # to save memory, load chunk_size captions at a time
    for s in tqdm(range(0, len(answers), chunk_size)):

        coco_res = coco.loadRes(answers[s: min(s+chunk_size, len(answers))])
        coco_eval = COCOEvalCap(coco, coco_res)
        
        coco_eval.params["image_id"] = coco_res.getImgIds()
        coco_eval.evaluate()

        for metric, score in coco_eval.eval.items():
            all_overall_scores[metric].append(score)
        
        for i, cur_img_id in enumerate(coco_res.getImgIds()):
            cur_eval_dict = coco_eval.evalImgs[i]
            # add caption to the eval dict
            cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
            img_to_eval_dict[cur_img_id] = cur_eval_dict

    # overall result
    overall_dict = {}
    for metric, score in all_overall_scores.items():
        overall_dict[metric] = np.mean(score)
    formulated_output_dict["overall"] = overall_dict
    formulated_output_dict["imgToEval"] = img_to_eval_dict

    chair_file = os.path.join(os.path.dirname(answer_file), "chair_" + os.path.basename(answer_file))
    json.dump(formulated_output_dict, open(chair_file, "w"))

    _, imids, _ = load_generated_captions(chair_file)

    evaluator = CHAIR(imids, anno_dir)
    evaluator.get_annotations()
    cap_dict = evaluator.compute_chair(chair_file)

    metrics_output_path = os.path.join(os.path.dirname(answer_file), "metrics_" + os.path.basename(answer_file))
    with open(metrics_output_path, "w") as f:
        json.dump(cap_dict, f, indent=4)
    print(f"Metrics saved to {metrics_output_path}")


def eval_pope(answers, labels, question_ids):
    pos_num = 0
    print(len(answers), len(labels), len(question_ids))
    pred_list, label_list, error_id = [], [], []
    # print(answers.keys())
    for question_id in question_ids:
        ### process answer
        # print(question_id)
        if question_id not in answers.keys(): continue
        text = answers[question_id]

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            pred_list.append(0)
        else:
            pred_list.append(1)

        ### process label
        if labels[question_id] and 'no' in labels[question_id].lower():
            label_list.append(0)
        else:
            label_list.append(1)
            pos_num += 1

        ## sta_error
        if pred_list[-1] != label_list[-1]:
            # print(question_id)
            error_id.append(question_id)

    # with open(error_file, 'w') as fw:
    #     json.dump(error_id, fw)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    assert len(pred_list) == len(label_list)
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\tTotal\t')
    print('{}\t{}\t{}\t{}\t{}'.format(TP, FP, TN, FN, TP + FP + TN + FN))

    precision = float(TP) / float(TP + FP + 0.00001)
    recall = float(TP) / float(TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.4f, %.4f, %.4f, %.4f, %.4f' % (acc, precision, recall, f1, yes_ratio))

    print('Total_num:', len(label_list))
    print('pos_num', pos_num, 'neg_num', len(label_list) - pos_num)

    return [acc, precision, recall, f1, yes_ratio]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--eval_pope", action='store_true', default=False)
    parser.add_argument("--eval_chair", action='store_true', default=False)
    parser.add_argument("--pope_savepath", type=str)
    args = parser.parse_args()

    if args.eval_chair:
        eval_chair(args.result_file)
    elif args.eval_pope:
        questions = [json.loads(d) for d in open(args.question_file, "r")]
        questions = {question['question_id']: question for question in questions}
        labels = [json.loads(d) for d in open(args.question_file, "r")]
        answers = [json.loads(d) for d in open(args.result_file, "r")]
        question_ids = list(questions.keys())
        print(len(list(set(question_ids))))
        answers_list = {a['question_id']: a['text'] for a in answers}
        label_list = {q['question_id']: q['label'] for q in labels}
        result = eval_pope(answers_list, label_list, question_ids)
        results = np.round(np.multiply(np.array(result), 100), decimals=2)
        print('Average:', results)
        
        pope_metric = {
            'raw_metrics': {
                'accuracy': result[0],
                'precision': result[1],
                'recall': result[2],
                'f1_score': result[3],
                'yes_ratio': result[4]
            },
            'Average_metrics': {
                'accuracy': results[0],
                'precision': results[1],
                'recall': results[2],
                'f1_score': results[3],
                'yes_ratio': results[4]
            }
        }

        os.makedirs(os.path.dirname(args.pope_savepath), exist_ok=True)
        with open(args.pope_savepath, 'w') as f:
            json.dump(pope_metric, f, indent=4)
    print("====================================")
