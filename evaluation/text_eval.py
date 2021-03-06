from .rrc_evaluation_funcs import validate_clockwise_points
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from evaluation import scripts as text_eval_script
import zipfile
from evaluation import geometry


class TextEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        # if not hasattr(self._metadata, "json_file"):
        #     raise AttributeError(
        #         f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
        #     )

        # json_file = PathManager.get_local_path(self._metadata.json_file)
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._coco_api = COCO(json_file)

        # use dataset_name to decide eval_gt_path

        if "icdar15" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/icdar15_gt.zip"
            self._word_spotting = False
        elif "icdar" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/icdar_gt.zip"
            self._word_spotting = False
        self._text_eval_confidence = 0

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_name": os.path.basename(input["file_name"])}

            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = instances_to_coco_json(
                instances, os.path.basename(input["file_name"]))
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results", cf_th=0.25):

        with open(file_path, 'r') as f:
            data = json.load(f)
            with open('temp_all_det_cors.txt', 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_name'])

                        for i in range(len(data[ix]['polys'])):
                            outstr = outstr + \
                                str(int(data[ix]['polys'][i][0])) + ',' + \
                                str(int(data[ix]['polys'][i][1])) + ','
                        outstr = outstr[0:-1]+"\n"
                        # outstr = outstr + \
                        #     str(round(data[ix]['score'], 3)) +'\n'
                        f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        lsc = [cf_th]
        fres = open('temp_all_det_cors.txt', 'r').readlines()
        for isc in lsc:
            if not os.path.isdir(dirn):
                os.mkdir(dirn)

            for line in fres:
                line = line.strip()
                s = line.split(': ')

                filename = '{}.txt'.format(s[0].split(".")[0])
                outName = os.path.join(dirn, filename)
                with open(outName, 'a') as fout:
                    fout.writelines(s[1]+'\n')

        os.remove("temp_all_det_cors.txt")

    def sort_detection(self, temp_dir):
        origin_file = temp_dir
        output_file = "final_"+temp_dir

        if not os.path.isdir(output_file):
            os.mkdir(output_file)

        files = glob.glob(origin_file+'*.txt')
        files.sort()

        for i in files:
            out = i.replace(origin_file, output_file)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')

            for iline, line in enumerate(fin):
                cors = line.strip().split(',')

                assert(len(cors) % 2 == 0), 'cors invalid.'
                pts = [(int(cors[j]), int(cors[j+1]))
                       for j in range(0, len(cors), 2)]
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print(
                        'An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue

                if not pgt.is_valid:
                    print(
                        'An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue

                pRing = LinearRing(pts)
                if pRing.is_ccw:
                    pts.reverse()
                outstr = ''
                for ipt in pts[:-1]:
                    outstr += (str(int(ipt[0]))+',' + str(int(ipt[1]))+',')
                outstr += (str(int(pts[-1][0]))+',' + str(int(pts[-1][1])))
                #outstr = outstr+','+str(score)
                fout.writelines(outstr+'\n')

            fout.close()
        os.chdir(output_file)

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        zipf = zipfile.ZipFile('../det.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('./', zipf)
        zipf.close()
        os.chdir("../")
        # clean temp files
        shutil.rmtree(origin_file)
        #shutil.rmtree(output_file)
        return "det.zip"

    def evaluate_with_official_code(self, result_path, gt_path):
        return text_eval_script.text_eval_main(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        coco_results = list(itertools.chain(
            *[x["instances"] for x in predictions]))
        PathManager.mkdirs(self._output_dir)

        file_path = os.path.join(self._output_dir, "text_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        self._results = OrderedDict()

        # eval text
        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir, self._text_eval_confidence)
        result_path = self.sort_detection(temp_dir)
        text_result = self.evaluate_with_official_code(
            result_path, self._text_eval_gt_path)
        os.remove(result_path)

        print(text_result["Message"])

        return copy.deepcopy(self._results)


def instances_to_coco_json(instances, image_name):

    results = list()
    num_instances = len(instances)
    if num_instances == 0:
        return []

    scores = instances.scores.tolist()
    rboxes = instances.rboxes

    for rbox, score in zip(rboxes, scores):

        format_rbox = geometry.sort_vertex8(rbox)
        try:
            validate_clockwise_points(format_rbox)
        except Exception as e:
            print(e, format_rbox)
            continue

        format_rbox = [[format_rbox[0], format_rbox[1]], [format_rbox[2], format_rbox[3]], [

            format_rbox[4], format_rbox[5]], [format_rbox[6], format_rbox[7]]]

        result = {
            "image_name": image_name,
            "category_id": 1,
            "polys": format_rbox,
            "score": score
        }
        results.append(result)

    return results
