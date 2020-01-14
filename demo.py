import numpy as np
import sys
import os

baseDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{0}/build'.format(baseDir))

#sys.path.insert(0, '/home/busta/git/caffe_orig/Release/install/python')
# sys.path.insert(0, '/mnt/textspotter/software/opencv/ReleaseStatic/lib')
sys.path.insert(0,'/home/innoruption/caffe/Release/install/python')

import caffe
import cv2
import os
import glob

import math
import time

from models import create_models_tiny

import vis

image_no = 0

from utils import get_normalized_image, print_seq_ext, print_seq2, get_obox, process_splits

# import cmp_trie

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from os import listdir
from PIL import Image as PImage

buckets = [54, 80, 124, 182, 272, 410, 614, 922, 1383, 2212]  #batch by rubs
image_size = [544, 544]
it = 0
mean_rec = 0
gt_all = 0
gt_loc = 0
wr_good_all = 0

debug = False

rec_t = 0.6
ext_factor = 1.1
ext_factorx = 1.3


def froward_image(nets, scaled, original):
    global rec_t, ext_factor, ext_factorx

    net, net_ctc = nets
    print("nets:",nets)
    print("net: ",net)
    print("net_ctc :",net_ctc)

    img = [scaled]
    draw = img[0]

    imgo = original

    im = np.asarray(img, dtype=np.float)
    im = im / 128.0
    im = im - 1.0
    # im = im.reshape((3, im.shape[0], im.shape[1]))
    im = np.swapaxes(im, 1, 3)
    im = np.swapaxes(im, 2, 3)

    net.blobs['data'].reshape(im.shape[0], im.shape[1], im.shape[2], im.shape[3])
    print("net.blobs['data'] :", net.blobs['data'])
    net.blobs['data'].data[...] = im
    print("im: ", im)
    net.reshape()
    print("net.reshape(): ", net.reshape())
    start = time.time()
    out = net.forward(start="conv1")
    end = time.time()
    seconds = end - start
    fps = 1 / seconds
    # print("loc fps:{0}".format(fps))

    boxes = out['boxes']
    print("boxes: ", boxes)
    boxes[0, 0, :, 0] *= image_size[0]
    boxes[0, 0, :, 1] *= image_size[1]
    normFactor = math.sqrt(image_size[1] * image_size[1] + image_size[0] * image_size[0])
    boxes[0, 0, :, 2] *= normFactor
    boxes[0, 0, :, 3] *= normFactor

    nms = boxes[0, 0, :, 8] != 1
    boxes = boxes[:, :, nms, :]
    print("boxes before boxes_count: ", boxes)
    boxes_count = 0
    for i in range(0, boxes.shape[2]):
        det_word = boxes[0, 0, i]
        if (det_word[0] == 0 and det_word[1] == 0) or det_word[5] < 0.1:
            break
        boxes_count += 1


    detections_out = []

    for i in range(0, boxes_count):
        det_word = boxes[0, 0, i]
        boxr = ((det_word[0], det_word[1]), (det_word[2], det_word[3]), det_word[4] * 180 / 3.14)
        print("boxr : this is r box,", boxr)
        box = cv2.boxPoints(boxr)
        print("box : sfter detection count,",box)

        box = np.array(box, dtype="int")
        vis.draw_box_points(draw, box, (255, 0, 0))
        bbox = cv2.boundingRect(box)
        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        boxro = [[det_word[0], det_word[1]], [det_word[2] * ext_factorx, det_word[3] * ext_factor],
                 det_word[4] * 180 / 3.14]
        boxt = get_obox(img[0], original, boxro)
        print("boxt 1 :",boxt)
        boxt = ((boxt[0][0], boxt[0][1]), (boxt[1][0], boxt[1][1]), boxt[2])
        print("boxt 2 :", boxt)

        norm2, rot_mat = get_normalized_image(original, boxt)
        if norm2 is None:
            continue

        norm = cv2.cvtColor(norm2, cv2.COLOR_BGR2GRAY)
        print("Given Norm :",norm)

        width_scale = 32.0 / norm2.shape[0]
        width = norm.shape[1] * width_scale
        best_diff = width
        bestb = 0
        for b in range(0, len(buckets)):
            if best_diff > abs(width - buckets[b]):
                best_diff = abs(width - buckets[b])
                bestb = b

        scaled = cv2.resize(norm, (buckets[bestb], 32))

        cv2.imshow('norm2', scaled)

        imtf = np.asarray([scaled], dtype=np.float)
        imtf = np.asarray(imtf, dtype=np.float)
        delta = imtf.max() - imtf.min()
        imtf /= (delta / 2)
        imtf -= imtf.mean()
        imtf = np.reshape(imtf, (imtf.shape[0], -1, imtf.shape[1], imtf.shape[2]))

        net_ctc.blobs['data'].reshape(imtf.shape[0], imtf.shape[1], imtf.shape[2], imtf.shape[3])
        net_ctc.blobs['data'].data[...] = imtf

        outctc = net_ctc.forward()
        print("outctc : ",outctc)
        ctc_f = outctc['softmax']
        print("ctc_f : ", ctc_f)

        ctc_f = ctc_f.reshape(ctc_f.shape[0], ctc_f.shape[1], ctc_f.shape[3])
        labels = ctc_f.argmax(2)
        mask = labels > 2
        masked = ctc_f.max(2)[mask]
        mean_conf = np.sum(masked) / masked.shape[0]
        print("mean_conf : ", mean_conf)

        if mean_conf < 0.2:
            vis.draw_box_points(scaled, box, color=(0, 0, 0))
            continue

        if debug:
            vis.vis_square(imtf[0])

        det_text, conf, dec_s = print_seq_ext(labels[:, 0], np.sum(masked))

        if len(det_text) == 0:
            continue

        if len(det_text) < 3 and mean_conf < 0.8:
            continue
        print("detections_out: ", detections_out)
        detections_out.append((boxt, (det_text, mean_conf, int(det_word[6]))))
        continue

        splits_raw = process_splits(det_text, conf, dec_s, norm2, ctc_f, rot_mat, boxt, original, 0, mean_conf,
                                    alow_non_dict=True)
        detections_out.extend(splits_raw)
        continue

        if out_raw is not None:
            out_raw.write(u"{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}\n".format( \
                'vid', box[0, 0], box[0, 1], box[1, 0], box[1, 1], \
                box[2, 0], box[2, 1], box[3, 0], box[3, 1], det_text, det_text, mean_conf).encode('utf8'))

        dec2, conf2, dec_splits = cmp_trie.decode_sofmax(ctc_f.reshape(ctc_f.shape[0], ctc_f.shape[2]))
        best_dict = print_seq2(dec2[0])

        if len(best_dict) == 0:
            continue
        splits_out = process_splits(best_dict, conf2, dec_splits, norm2, ctc_f, rot_mat, boxt, original, 1, mean_conf)
        detections_out.extend(splits_out)

    return detections_out, fps


path = "/home/innoruption/Rubseena/DeepTextSpotter/src/images/"
#path="/home/innoruption/Rubseena/DeepTextSpotter/test/"


# your images in an array
# imgs = loadImages(path)

# for img in imgs:
# you can show every image
# img.show()


def test_image(nets):
    img_dir = "src/images"
    #img_dir = "test"

    imgs = [os.path.join(path, f) for f in os.listdir(img_dir)]
    for img in imgs:
        im = cv2.imread(img)
        global rec_t, image_size
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf", 16)
        font2 = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf", 18)
        image_size = [640 / 64 * 64, 480 / 64 * 64]

        scaled = cv2.resize(im, (image_size[0], image_size[1]))
        if nets[0].blobs['data'].data[...].shape[1] == 1:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            scaled = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))

        detections_out, fps = froward_image(nets, scaled, im)

        img = Image.fromarray(im)
        draw = ImageDraw.Draw(img)

        text_list = list()
        for detection in detections_out:
            print("Detection",detection)
            text = detection[1][0]
            text = text.encode('ascii', 'ignore')
            text_list.append(text)
            print(text_list)

            width, height = draw.textsize(text, font=font)
            center = [detection[0][0][0] - width / 2, detection[0][0][1] - 10]

            sx = int(detection[0][0][0] - width / 2)
            ex = int(detection[0][0][0] + width / 2)
            sy = int(detection[0][0][1] - 10)
            ey = int(detection[0][0][1] + 10)

            im[sy:ey, sx:ex] = im[sy:ey, sx:ex] / 2

            boxr = ((detection[0][0][0], detection[0][0][1]), (detection[0][1][0], detection[0][1][1]), detection[0][2])
            box = cv2.boxPoints(boxr)
            color = (0, 255, 0)
            vis.draw_box_points(im, box, color, thickness=1)

        img = Image.fromarray(im)
        draw = ImageDraw.Draw(img)

        draw.text((10, 10), 'FPS: {0:.2f}'.format(fps), (0, 255, 0), font=font2)

        draw.text((image_size[1] / 2 - 150, image_size[0] / 2 - 100), 'Raw Detections with Dictionary', (0, 0, 255),
                  font=font2)
        print('text detectionnnnnnnnnnnnnn', text_list)

        print( text_list[0])
        center_list = list()
        center_text = list()
        for detection in detections_out:
            text = detection[1][0]

            width, height = draw.textsize(text, font=font)
            center = [detection[0][0][0] - width / 2, detection[0][0][1] - 10]
            print("center",center)
            center_list.append(center)
            print("text: ", text)
            print("center list",center_list)


            draw.text((center[0], center[1]), text, fill=(0, 255, 0), font=font)
        sorted_center = sorted(center_list, key=lambda x: (x[0], x[1]))
        print("sorted_center", sorted_center)
        pix = np.array(img)

        cv2.imshow('draw', scaled)
        #
        if pix.shape[0] > 1024:
            pix = cv2.resize(pix, (pix.shape[1] / 2, pix.shape[0] / 2))
        cv2.imshow('pix', pix)
        cv2.waitKey(0)



def test_video(nets):
    global rec_t, image_size
    cap = cv2.VideoCapture('/mnt/textspotter/evaluation-sets/icdar2013-video-Test/Video_35_2_3.mp4')
    cap = cv2.VideoCapture(-1)
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf", 16)
    font2 = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf", 18)

    ret, im = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('/tmp/output.avi', fourcc, 20.0, (im.shape[1], im.shape[0]))

    frame_no = 0
    while ret:
        image_size = [640 / 64 * 64, 480 / 64 * 64]
        ret, im = cap.read()

        if ret == True:

            scaled = cv2.resize(im, (image_size[0], image_size[1]))
            if nets[0].blobs['data'].data[...].shape[1] == 1:
                scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                scaled = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))

            detections_out, fps = froward_image(nets, scaled, im)

            img = Image.fromarray(im)
            draw = ImageDraw.Draw(img)

            for detection in detections_out:
                text = detection[1][0]
                print(text)
                width, height = draw.textsize(text, font=font)
                center = [detection[0][0][0] - width / 2, detection[0][0][1] - 10]

                sx = int(detection[0][0][0] - width / 2)
                ex = int(detection[0][0][0] + width / 2)
                sy = int(detection[0][0][1] - 10)
                ey = int(detection[0][0][1] + 10)

                im[sy:ey, sx:ex] = im[sy:ey, sx:ex] / 2

                boxr = (
                (detection[0][0][0], detection[0][0][1]), (detection[0][1][0], detection[0][1][1]), detection[0][2])
                box = cv2.boxPoints(boxr)
                color = (0, 255, 0)
                vis.draw_box_points(im, box, color, thickness=1)

            img = Image.fromarray(im)
            draw = ImageDraw.Draw(img)

            draw.text((10, 10), 'FPS: {0:.2f}'.format(fps), (0, 255, 0), font=font2)
            frame_no += 1

            if frame_no < 30:
                draw.text((image_size[1] / 2 - 150, image_size[0] / 2 - 100), 'Raw Detections with Dictionary',
                          (0, 0, 255), font=font2)

            for detection in detections_out:
                text = detection[1][0]
                width, height = draw.textsize(text, font=font)
                center = [detection[0][0][0] - width / 2, detection[0][0][1] - 10]
                draw.text((center[0], center[1]), text, fill=(0, 255, 0), font=font)

            pix = np.array(img)

            cv2.imshow('draw', scaled)
            #
            if pix.shape[0] > 1024:
                pix = cv2.resize(pix, (pix.shape[1] / 2, pix.shape[0] / 2))
            cv2.imshow('pix', pix)

            out.write(pix)
            cv2.waitKey(10)

    out.release()


if __name__ == '__main__':
    # caffe.set_mode_gpu()
    nets = create_models_tiny(caffe.TEST)
    print(nets)
    yolonet = nets[0]
    net_ctc = nets[1]

    #test_video(nets)
    test_image(nets)
    # your images in an array
    # loadImages(path)
