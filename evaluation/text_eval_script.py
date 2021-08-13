from collections import namedtuple
from evaluation import rrc_evaluation_funcs
import importlib
import sys
import numpy as np

import math

from rapidfuzz import string_metric

WORD_SPOTTING = True


def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """
    return {
        'Polygon': 'plg',
        'numpy': 'np'
    }


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    global WORD_SPOTTING
    return {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'WORD_SPOTTING': WORD_SPOTTING,
        'MIN_LENGTH_CARE_WORD': 3,
        'GT_SAMPLE_NAME_2_ID': '([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': '([0-9]+).txt',
        # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'LTRB': False,
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        # Detections must include confidence value. MAP and MAR will be calculated,
        'CONFIDENCES': False,
        'SPECIAL_CHARACTERS': str('!?.:,*"()·[]/\''),
        'ONLY_REMOVE_FIRST_LAST_CHARACTER': True
    }


def validate_data(gt_file_path, subm_file_path, evaluation_params):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(
        gt_file_path, evaluation_params['GT_SAMPLE_NAME_2_ID'])

    
    subm = rrc_evaluation_funcs.load_zip_file(
        subm_file_path, evaluation_params['DET_SAMPLE_NAME_2_ID'], True)
    

    # Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file_gt(
            k, gt[k], evaluation_params['CRLF'], evaluation_params['LTRB'], True)

    # Validate format of results
    subm=dict([(str(int(k)),v) for k,v in subm.items()])
    
    for k in subm:
        if (k in gt) == False:
            raise Exception("The sample %s not present in GT" % k)

        rrc_evaluation_funcs.validate_lines_in_file(
            k, subm[k], evaluation_params['CRLF'], evaluation_params['LTRB'], True, evaluation_params['CONFIDENCES'])


def evaluate_method(gt_file_path, subm_file_path, evaluation_params):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """
    for module, alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        num_points = len(points)
        # res_boxes=np.empty([1,num_points],dtype='int32')
        res_boxes = np.empty([1, num_points], dtype='float32')
        for inp in range(0, num_points, 2):
            res_boxes[0, int(inp/2)] = float(points[int(inp)])
            res_boxes[0, int(inp/2+num_points/2)] = float(points[int(inp+1)])
        point_mat = res_boxes[0].reshape([2, int(num_points/2)]).T
        return plg.Polygon(point_mat)

    def rectangle_to_polygon(rect):
        res_boxes = np.empty([1, 8], dtype='int32')
        res_boxes[0, 0] = int(rect.xmin)
        res_boxes[0, 4] = int(rect.ymax)
        res_boxes[0, 1] = int(rect.xmin)
        res_boxes[0, 5] = int(rect.ymin)
        res_boxes[0, 2] = int(rect.xmax)
        res_boxes[0, 6] = int(rect.ymin)
        res_boxes[0, 3] = int(rect.xmax)
        res_boxes[0, 7] = int(rect.ymax)

        point_mat = res_boxes[0].reshape([2, 4]).T

        return plg.Polygon(point_mat)

    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(
            rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points

    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def compute_ap(conf_list, match_list, num_gt_care):
        correct = 0
        AP = 0
        if len(conf_list) > 0:
            conf_list = np.array(conf_list)
            match_list = np.array(match_list)
            sorted_ind = np.argsort(-conf_list)
            conf_list = conf_list[sorted_ind]
            match_list = match_list[sorted_ind]
            for n in range(len(conf_list)):
                match = match_list[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)

            if num_gt_care > 0:
                AP /= num_gt_care

        return AP

    def transcription_match(trans_gt, trans_det, special_characters=str(r'!?.:,*"()·[]/\''), only_remove_first_last_character_gt=True):

        if only_remove_first_last_character_gt:
            # special characters in GT are allowed only at initial or final position
            if (trans_gt == trans_det):
                return True

            if special_characters.find(trans_gt[0]) > -1:
                if trans_gt[1:] == trans_det:
                    return True

            if special_characters.find(trans_gt[-1]) > -1:
                if trans_gt[0:len(trans_gt)-1] == trans_det:
                    return True

            if special_characters.find(trans_gt[0]) > -1 and special_characters.find(trans_gt[-1]) > -1:
                if trans_gt[1:len(trans_gt)-1] == trans_det:
                    return True
            return False
        else:
            # Special characters are removed from the begining and the end of both Detection and GroundTruth
            while len(trans_gt) > 0 and special_characters.find(trans_gt[0]) > -1:
                trans_gt = trans_gt[1:]

            while len(trans_det) > 0 and special_characters.find(trans_det[0]) > -1:
                trans_det = trans_det[1:]

            while len(trans_gt) > 0 and special_characters.find(trans_gt[-1]) > -1:
                trans_gt = trans_gt[0:len(trans_gt)-1]

            while len(trans_det) > 0 and special_characters.find(trans_det[-1]) > -1:
                trans_det = trans_det[0:len(trans_det)-1]

            return trans_gt == trans_det

    def include_in_dictionary(transcription):
        """
        Function used in Word Spotting that finds if the Ground Truth transcription meets the rules to enter into the dictionary. If not, the transcription will be cared as don't care
        """
        # special case 's at final
        if transcription[len(transcription)-2:] == "'s" or transcription[len(transcription)-2:] == "'S":
            transcription = transcription[0:len(transcription)-2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        special_characters = str("'!?.:,*\"()·[]/")
        for character in special_characters:
            transcription = transcription.replace(character, ' ')

        transcription = transcription.strip()

        if len(transcription) != len(transcription.replace(" ", "")):
            return False

        if len(transcription) < evaluation_params['MIN_LENGTH_CARE_WORD']:
            return False

        not_allowed = str("×÷·")

        range1 = [ord(u'a'), ord(u'z')]
        range2 = [ord(u'A'), ord(u'Z')]
        range3 = [ord(u'À'), ord(u'ƿ')]
        range4 = [ord(u'Ǆ'), ord(u'ɿ')]
        range5 = [ord(u'Ά'), ord(u'Ͽ')]
        range6 = [ord(u'-'), ord(u'-')]

        for char in transcription:
            char_code = ord(char)
            if(not_allowed.find(char) != -1):
                return False

            valid = (char_code >= range1[0] and char_code <= range1[1]) or (char_code >= range2[0] and char_code <= range2[1]) or (char_code >= range3[0] and char_code <= range3[1]) or (
                char_code >= range4[0] and char_code <= range4[1]) or (char_code >= range5[0] and char_code <= range5[1]) or (char_code >= range6[0] and char_code <= range6[1])
            if valid == False:
                return False

        return True

    def include_in_dictionary_transcription(transcription):
        """
        Function applied to the Ground Truth transcriptions used in Word Spotting. It removes special characters or terminations
        """
        # special case 's at final
        if transcription[len(transcription)-2:] == "'s" or transcription[len(transcription)-2:] == "'S":
            transcription = transcription[0:len(transcription)-2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        special_characters = str("'!?.:,*\"()·[]/")
        for character in special_characters:
            transcription = transcription.replace(character, ' ')

        transcription = transcription.strip()

        return transcription

    per_sample_metrics = {}

    matched_sum = 0
    det_only_matched_sum = 0

    rectangle = namedtuple('rectangle', 'xmin ymin xmax ymax')

    gt = rrc_evaluation_funcs.load_zip_file(
        gt_file_path, evaluation_params['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(
        subm_file_path, evaluation_params['DET_SAMPLE_NAME_2_ID'], True)

    num_global_care_gt = 0
    num_global_care_det = 0
    det_only_num_global_care_gt = 0
    det_only_num_global_care_det = 0

    arrGlobalConfidences = []
    arrGlobalMatches = []

    for res_file in gt:

        gtFile = rrc_evaluation_funcs.decode_utf8(gt[res_file])
        if (gtFile is None):
            raise Exception("The file %s is not UTF-8" % res_file)

        recall = 0
        precision = 0
        hmean = 0
        det_correct = 0
        det_only_correct = 0
        iou_mat = np.empty([1, 1])
        gt_pols = []
        det_pols = []
        gt_trans = []
        det_trans = []
        gt_pol_points = []
        det_pol_points = []
        gt_dont_care_pols_num = []  # Array of Ground Truth Polygons' keys marked as don't Care
        det_only_gt_dont_care_pols_num = []
        det_dont_care_pols_num = []  # Array of Detected Polygons' matched with a don't Care GT
        det_only_det_dont_care_pols_num = []
        det_matched_nums = []
        pairs = []

        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        points_list, _, transcriptions_list = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
            gtFile, evaluation_params['CRLF'], evaluation_params['LTRB'], True, False)


        for n in range(len(points_list)):
            points = points_list[n]
            transcription = transcriptions_list[n]
            # ctw1500 and total_text gt have been modified to the same format.
            det_only_dont_care = dont_care = transcription == "###"
            if evaluation_params['LTRB']:
                gt_rect = rectangle(*points)
                gt_pol = rectangle_to_polygon(gt_rect)
            else:
                gt_pol = polygon_from_points(points)
            gt_pols.append(gt_pol)
            gt_pol_points.append(points)

            # On word spotting we will filter some transcriptions with special characters
            if evaluation_params['WORD_SPOTTING']:
                if dont_care == False:
                    if include_in_dictionary(transcription) == False:
                        dont_care = True
                    else:
                        transcription = include_in_dictionary_transcription(
                            transcription)

            gt_trans.append(transcription)
            if dont_care:
                gt_dont_care_pols_num.append(len(gt_pols)-1)
            if det_only_dont_care:
                det_only_gt_dont_care_pols_num.append(len(gt_pols)-1)
        res_file="00"+res_file
        
        if res_file in subm:

            detFile = rrc_evaluation_funcs.decode_utf8(subm[res_file])

            points_list, confidences_list, transcriptions_list = rrc_evaluation_funcs.get_tl_line_values_from_file_contents_det(
                detFile, evaluation_params['CRLF'], evaluation_params['LTRB'], True, evaluation_params['CONFIDENCES'])
            
            
            for n in range(len(points_list)):
                points = points_list[n]
                transcription = transcriptions_list[n]

                if evaluation_params['LTRB']:
                    detRect = rectangle(*points)
                    det_pol = rectangle_to_polygon(detRect)
                else:
                    det_pol = polygon_from_points(points)
                det_pols.append(det_pol)
                det_pol_points.append(points)
                det_trans.append(transcription)

                if len(gt_dont_care_pols_num) > 0:
                    for dont_care_pol in gt_dont_care_pols_num:
                        dont_care_pol = gt_pols[dont_care_pol]
                        intersected_area = get_intersection(
                            dont_care_pol, det_pol)
                        pd_dimensions = det_pol.area()
                        precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                        if (precision > evaluation_params['AREA_PRECISION_CONSTRAINT']):
                            det_dont_care_pols_num.append(len(det_pols)-1)
                            break

                if len(det_only_gt_dont_care_pols_num) > 0:
                    for dont_care_pol in det_only_gt_dont_care_pols_num:
                        dont_care_pol = gt_pols[dont_care_pol]
                        intersected_area = get_intersection(
                            dont_care_pol, det_pol)
                        pd_dimensions = det_pol.area()
                        precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                        if (precision > evaluation_params['AREA_PRECISION_CONSTRAINT']):
                            det_only_det_dont_care_pols_num.append(
                                len(det_pols)-1)
                            break

            
            if len(gt_pols) > 0 and len(det_pols) > 0:
                # Calculate IoU and precision matrixs
                output_shape = [len(gt_pols), len(det_pols)]
                iou_mat = np.empty(output_shape)
                gt_rect_mat = np.zeros(len(gt_pols), np.int8)
                det_rect_mat = np.zeros(len(det_pols), np.int8)
                det_only_gt_rect_mat = np.zeros(len(gt_pols), np.int8)
                det_only_det_rect_mat = np.zeros(len(det_pols), np.int8)
                for gt_num in range(len(gt_pols)):
                    for det_num in range(len(det_pols)):
                        pG = gt_pols[gt_num]
                        pD = det_pols[det_num]
                        iou_mat[gt_num, det_num] = get_intersection_over_union(
                            pD, pG)

                for gt_num in range(len(gt_pols)):
                    for det_num in range(len(det_pols)):
                        if gt_rect_mat[gt_num] == 0 and det_rect_mat[det_num] == 0 and gt_num not in gt_dont_care_pols_num and det_num not in det_dont_care_pols_num:
                            if iou_mat[gt_num, det_num] > evaluation_params['IOU_CONSTRAINT']:
                                gt_rect_mat[gt_num] = 1
                                det_rect_mat[det_num] = 1
                                # detection matched only if transcription is equal
                                # det_only_correct = True
                                # det_only_correct += 1
                                if evaluation_params['WORD_SPOTTING']:
                                    edd = string_metric.levenshtein(
                                        gt_trans[gt_num].upper(), det_trans[det_num].upper())
                                    if edd <= 0:
                                        correct = True
                                    else:
                                        correct = False
                                    # correct = gt_trans[gt_num].upper() == det_trans[det_num].upper()
                                else:
                                    try:
                                        correct = transcription_match(gt_trans[gt_num].upper(), det_trans[det_num].upper(
                                        ), evaluation_params['SPECIAL_CHARACTERS'], evaluation_params['ONLY_REMOVE_FIRST_LAST_CHARACTER']) == True
                                    except:  # empty
                                        correct = False
                                det_correct += (1 if correct else 0)
                                if correct:
                                    det_matched_nums.append(det_num)

                for gt_num in range(len(gt_pols)):
                    for det_num in range(len(det_pols)):
                        if det_only_gt_rect_mat[gt_num] == 0 and det_only_det_rect_mat[det_num] == 0 and gt_num not in det_only_gt_dont_care_pols_num and det_num not in det_only_det_dont_care_pols_num:
                            if iou_mat[gt_num, det_num] > evaluation_params['IOU_CONSTRAINT']:
                                det_only_gt_rect_mat[gt_num] = 1
                                det_only_det_rect_mat[det_num] = 1
                                # detection matched only if transcription is equal
                                det_only_correct = True
                                det_only_correct += 1

        num_gt_care = (len(gt_pols) - len(gt_dont_care_pols_num))
        num_det_care = (len(det_pols) - len(det_dont_care_pols_num))
        det_only_num_gt_care = (
            len(gt_pols) - len(det_only_gt_dont_care_pols_num))
        det_only_num_det_care = (
            len(det_pols) - len(det_only_det_dont_care_pols_num))
        if num_gt_care == 0:
            recall = float(1)
            precision = float(0) if num_det_care > 0 else float(1)
        else:
            recall = float(det_correct) / num_gt_care
            precision = 0 if num_det_care == 0 else float(
                det_correct) / num_det_care

        if det_only_num_gt_care == 0:
            det_only_recall = float(1)
            det_only_precision = float(
                0) if det_only_num_det_care > 0 else float(1)
        else:
            det_only_recall = float(det_only_correct) / det_only_num_gt_care
            det_only_precision = 0 if det_only_num_det_care == 0 else float(
                det_only_correct) / det_only_num_det_care

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
            precision * recall / (precision + recall)
        det_only_hmean = 0 if (det_only_precision + det_only_recall) == 0 else 2.0 * \
            det_only_precision * det_only_recall / \
            (det_only_precision + det_only_recall)

        matched_sum += det_correct
        det_only_matched_sum += det_only_correct
        num_global_care_gt += num_gt_care
        num_global_care_det += num_det_care
        det_only_num_global_care_gt += det_only_num_gt_care
        det_only_num_global_care_det += det_only_num_det_care

        per_sample_metrics[res_file] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'iou_mat': [] if len(det_pols) > 100 else iou_mat.tolist(),
            'gt_pol_points': gt_pol_points,
            'det_pol_points': det_pol_points,
            'gt_trans': gt_trans,
            'det_trans': det_trans,
            'gt_dont_care': gt_dont_care_pols_num,
            'det_dont_care': det_dont_care_pols_num,
            'evaluation_params': evaluation_params,
        }

    method_recall = 0 if num_global_care_gt == 0 else float(
        matched_sum)/num_global_care_gt
    method_precision = 0 if num_global_care_det == 0 else float(
        matched_sum)/num_global_care_det
    method_hmean = 0 if method_recall + method_precision == 0 else 2 * \
        method_recall * method_precision / (method_recall + method_precision)

    det_only_method_recall = 0 if det_only_num_global_care_gt == 0 else float(
        det_only_matched_sum)/det_only_num_global_care_gt
    det_only_method_precision = 0 if det_only_num_global_care_det == 0 else float(
        det_only_matched_sum)/det_only_num_global_care_det
    det_only_method_hmean = 0 if det_only_method_recall + det_only_method_precision == 0 else 2 * \
        det_only_method_recall * det_only_method_precision / \
        (det_only_method_recall + det_only_method_precision)

    method_metrics = r"E2E_RESULTS: precision: {}, recall: {}, hmean: {}".format(
        method_precision, method_recall, method_hmean)
    det_only_method_metrics = r"DETECTION_ONLY_RESULTS: precision: {}, recall: {}, hmean: {}".format(
        det_only_method_precision, det_only_method_recall, det_only_method_hmean)

    res_dict = {'calculated': True, 'Message': '', 'e2e_method': method_metrics,
                'det_only_method': det_only_method_metrics, 'per_sample': per_sample_metrics}

    return res_dict


def text_eval_main(det_file, gt_file, is_word_spotting):
    global WORD_SPOTTING
    WORD_SPOTTING = is_word_spotting
    return rrc_evaluation_funcs.main_evaluation(None, det_file, gt_file, default_evaluation_params, validate_data, evaluate_method)
