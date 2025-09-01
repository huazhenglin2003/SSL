import os
import json
import random
from PIL import Image, ImageFile
from pycocotools.coco import COCO
from eval.chair import CHAIR


ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset:
    data_path = ""
    img_dir = ""

    def __init__(self, index_qid=True):
        self.index_qid = index_qid
    
    @property
    def all(self):
        return list(self.all_data_dict.keys())

    def process_data(self, data):
        img = Image.open(os.path.join(self.img_dir, data['image']))
        qu, gt = data['text'], data['label']
        return img, qu, gt

    def __getitem__(self, idx):
        if not self.index_qid:
            idx = self.all[idx] # get the qid
        data = self.all_data_dict[idx]
        return self.process_data(data)

    def __len__(self):
        return len(self.all_data_dict)
    
    
class POPE(BaseDataset):
    # TODO: modify the data path if you need
    data_path = "/data/pope/coco/coco_{}.json"
    img_dir = "/data/coco/val2014"

    def __init__(self, subset="pope_popular", **kwargs):
        super().__init__(**kwargs)
        all_data = [json.loads(d) for d in open(self.data_path.format(subset), 'r')]
        self.all_data_dict = {d['question_id']: d for d in all_data}
    

class CHAIRBench:
    # TODO: modify the data path if you need
    img_dir = "/data/coco/val2014"
    annot_file = "/data/coco/annotations/captions_val2014.json"
    annot_dir = "/data/coco/annotations"

    def __init__(self, num_samples=500, index_qid=True):
        self.index_qid = index_qid
        self.coco = COCO(self.annot_file)
        self.all = random.sample(self.coco.getImgIds(), num_samples)
        self._evaluator = None

    @property
    def evaluator(self):
        if self._evaluator is None:
            self._evaluator = CHAIR(self.all, self.annot_dir)
            self._evaluator.get_annotations()
        return self._evaluator

    def __getitem__(self, idx):
        if not self.index_qid:
            idx = self.all[idx] # get the qid
        image = self.coco.loadImgs(idx)[0]["file_name"]
        img = Image.open(os.path.join(self.img_dir, image))
        qu = "Please describe this image in detail."
        return img, qu, None

    def __len__(self):
        return len(self.all)
    
    def eval_hal(self, idx, gen_caption):
        imid = idx if self.index_qid else self.all[idx]
        gt_objects = self.evaluator.imid_to_objects[imid]
        words, node_words, _, _ = self.evaluator.caption_to_words(gen_caption)
        hallucinated_words = []
        for word, node_word in zip(words, node_words):
            if node_word not in gt_objects:
                hallucinated_words.append((node_word, word))
        if len(hallucinated_words) > 0:
            return True
        else:
            return False
    
    def get_words(self, gen_caption):
        words, _, _, _ = self.evaluator.caption_to_words(gen_caption)
        return words
    
    def eval_hal_words(self, idx, gen_caption):
        imid = idx if self.index_qid else self.all[idx]
        gt_objects = self.evaluator.imid_to_objects[imid]
        words, node_words, _, _ = self.evaluator.caption_to_words(gen_caption)
        return gt_objects, node_words
