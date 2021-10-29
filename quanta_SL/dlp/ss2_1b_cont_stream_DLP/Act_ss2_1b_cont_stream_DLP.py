###########################################################################
# Arin Can Ulku
# arin.ulku@epfl.ch
# EPFL, 2019
###########################################################################
import json
import math
import os
import os.path
import pathlib
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from natsort import natsorted

from ALP4 import ALP4
import time


def connectUI(obj):
    obj.ss2_1b_cont_stream_DLP.ui.visualize_data.clicked.connect(lambda: print_img(obj))
    obj.ss2_1b_cont_stream_DLP.ui.start_acq.clicked.connect(lambda: start_acq(obj))
    obj.ss2_1b_cont_stream_DLP.ui.browse_button.clicked.connect(
        lambda: browse_patterns(obj)
    )


def start_acq(obj):
    print("start_camera")
    # data_packs_size=1024*1024*4
    data_packs_size = 1024 * 1024 * 8
    data_chunk = 1024 * 1024 * 1024
    capture_time = float(obj.ss2_1b_cont_stream_DLP.ui.capture_time.text())
    frame_period = float(obj.ss2_1b_cont_stream_DLP.ui.frame_period.text())
    column_en = int(obj.ss2_1b_cont_stream_DLP.ui.column_en.text(), 16)
    row_start = int(obj.ss2_1b_cont_stream_DLP.ui.row_start.text())
    row_end = int(obj.ss2_1b_cont_stream_DLP.ui.row_end.text())
    gate_length = float(obj.ss2_1b_cont_stream_DLP.ui.gate_length.text())
    cont_acq = 0
    test_pattern = int(obj.ss2_1b_cont_stream_DLP.ui.debug_mode.isChecked())
    limited_gate_mode = int(obj.ss2_1b_cont_stream_DLP.ui.limited_gate_mode.isChecked())
    use_ext_trg = int(obj.ss2_1b_cont_stream_DLP.ui.use_ext_trg.isChecked())
    cont_chip_ctrl = 0
    op_mode = obj.ss2_1b_cont_stream_DLP.ui.op_mode.currentText()
    print("gate length", int(gate_length / float(5e-9)))
    print("frame_period", int(frame_period / float(5e-9)))
    print("column_en", column_en)
    print("row_start", row_start)
    print("row_end", row_end)
    print("limited_gate_mode", limited_gate_mode)
    if cont_acq:
        no_frames = int(capture_time / float(10.24e-6))
    else:
        no_frames = int(capture_time / frame_period)
    print("no_frames", no_frames)

    filename = str(obj.ss2_1b_cont_stream_DLP.ui.filename.text())
    file_path = os.path.join(str(pathlib.Path().absolute()), "usrsrc")
    file_path = os.path.join(file_path, "ss2_1b_cont_stream_DLP")
    file_path = os.path.join(
        file_path, str(obj.ss2_1b_cont_stream_DLP.ui.file_path.text())
    )

    column_en_str = obj.ss2_1b_cont_stream_DLP.ui.column_en.text()
    column_en_int = int(column_en_str, 16)
    column_en_bin = bin(column_en_int)
    read_size = (((row_end - row_start + 1) * 64 * column_en_bin.count("1"))) / 8
    total_read_size = int(no_frames) * read_size
    ceiled_read_size = math.ceil(total_read_size / data_chunk) * data_chunk
    print("ceiled_read_size", ceiled_read_size)
    usb_calls = int(int(ceiled_read_size / data_packs_size) / 128) * 128
    if usb_calls < 128:
        usb_calls = 128
    print("usb_calls", usb_calls)
    if os.path.exists(
        str(file_path)
        + "/"
        + str(((usb_calls - 1) >> 11))
        + "/"
        + str(((usb_calls - 1) >> 6))
    ):
        print("directory exists")
    else:
        print(str(file_path))
        if not os.path.exists(str(file_path)):
            os.makedirs(str(file_path))
        for x_dirs in range(0, usb_calls, 2048):
            if not os.path.exists(
                str(file_path) + "/" + str(int(x_dirs >> 11)) + str("/")
            ):
                os.makedirs(str(file_path) + "/" + str(int(x_dirs >> 11)) + str("/"))
        for x_dirs in range(0, usb_calls, 64):
            print(x_dirs)
            print(
                str(file_path)
                + str("/")
                + str(int(x_dirs >> 11))
                + str("/")
                + str(int(x_dirs >> 6))
                + str("/")
            )
            if not os.path.exists(
                str(file_path)
                + str("/")
                + str(int(x_dirs >> 11))
                + str("/")
                + str(int(x_dirs >> 6))
                + str("/")
            ):
                os.makedirs(
                    str(file_path)
                    + str("/")
                    + str(int(x_dirs >> 11))
                    + str("/")
                    + str(int(x_dirs >> 6))
                    + str("/")
                )
        print("directory created")
    file_path = file_path.replace("\\", "/")
    exe_path = os.path.join(os.path.abspath("."), "usrsrc")
    exe_path = os.path.join(exe_path, "ss2_1b_cont_stream_visualize")
    print(exe_path)
    argument = (
        exe_path
        + "\\ContinuousStreaming.exe "
        + exe_path
        + "\\210507_1_v2_ss2_top.bit"
        + " "
        + str(capture_time)
        + " "
        + str(frame_period)
        + " "
        + str(column_en)
        + " "
        + str(row_start)
        + " "
        + str(row_end)
        + " "
        + str(gate_length)
        + " "
        + str(test_pattern)
        + " "
        + str(filename)
        + " "
        + str(file_path)
        + " "
        + str(use_ext_trg)
        + " "
        + '"'
        + str(op_mode)
        + '"'
    )
    print(argument)
    # Load images
    pattern_folder = Path(obj.ss2_1b_cont_stream_DLP.ui.pattern_line.text())

    # in seconds
    project_frame_time = float(obj.ss2_1b_cont_stream_DLP.ui.pattern_time_line.text())

    img_file_list = [str(fpath) for fpath in pattern_folder.glob("*.png")]
    img_file_list += [str(fpath) for fpath in pattern_folder.glob("*.tiff")]
    img_file_list = natsorted(img_file_list)
    print("Found %d patterns." % len(img_file_list))
    if len(img_file_list) == 0:
        print("Finished.")
        return
    img_seq = [cv2.imread(fpath, -1) for fpath in img_file_list]

    # Repeat first frame
    num_buffer = 5
    img_first = [img_seq[0].ravel()] * num_buffer

    # Interleave repeat
    num_repeat = 1
    img_seq_except_first = img_seq[1:]
    img_seq_except_first = [
        frame.ravel() for frame in img_seq_except_first for _ in range(num_repeat)
    ]

    # Concatenate
    img_seq = [*img_first, *img_seq_except_first]
    num_patterns = len(img_seq)
    print(f"Num patterns after buffering and repeating {num_patterns}")
    img_seq = np.concatenate(img_seq)

    # Save metadata
    initial_sleep = 0.05
    metadata = {
        "capture_time": capture_time,
        "frame_period": frame_period,
        "patterns_folder": str(pattern_folder),
        "num_patterns": num_patterns,
        "project_frame_time": project_frame_time,
        "initial_sleep": initial_sleep,
    }
    with open(file_path + "metadata.json", "w") as fout:
        json.dump(metadata, fout, indent=4)

    # Load the Vialux .dll
    DMD = ALP4(version="4.2")
    # Initialize the device
    DMD.Initialize()

    print(f"DMD size {(DMD.nSizeY,DMD.nSizeX)}")

    # Allocate the onboard memory for the image sequence
    DMD.SeqAlloc(nbImg=num_patterns, bitDepth=1)
    # Send the image sequence as a 1D list/array/numpy array
    DMD.SeqPut(imgData=img_seq)

    # Set image rate to 50 Hz
    # in microseconds
    DMD.SetTiming(pictureTime=int(1e6 * project_frame_time))

    # Call cont-stream exe
    p = subprocess.Popen(r"%s" % argument)

    # Takes a little while before SPAD actually captures
    time.sleep(initial_sleep)

    # Project
    DMD.Run()

    # Wait until cont-stream finishes and exit
    p.wait()

    # Close the DMD
    # DMD.Wait()
    # Stop the sequence display
    DMD.Halt()
    # Free the sequence from the onboard memory
    DMD.FreeSeq()
    # De-allocate the device
    DMD.Free()

    print("Finished.")


def print_img(obj):
    # data_packs_size=1048576*4
    data_packs_size = 1048576 * 8
    data_chunk = 1024 * 1024 * 1024
    capture_time = float(obj.ss2_1b_cont_stream_DLP.ui.capture_time.text())
    frame_period = float(obj.ss2_1b_cont_stream_DLP.ui.frame_period.text())
    no_frames = int(capture_time / frame_period)
    row_end = int(obj.ss2_1b_cont_stream_DLP.ui.row_end.text())
    row_start = int(obj.ss2_1b_cont_stream_DLP.ui.row_start.text())
    usb_call_init = int(obj.ss2_1b_cont_stream_DLP.ui.usb_call_init.text())
    bits_size = int(obj.ss2_1b_cont_stream_DLP.ui.bits_size.text())
    stop_frame = int(obj.ss2_1b_cont_stream_DLP.ui.stop_frame.text())
    column_en_str = obj.ss2_1b_cont_stream_DLP.ui.column_en.text()
    column_en_int = int(column_en_str, 16)
    column_en = bin(column_en_int)
    column_en_bin = format(column_en_int, "08b")
    read_size = (((row_end - row_start + 1) * 64 * column_en.count("1"))) / 8
    total_read_size = int(no_frames) * read_size
    ceiled_read_size = math.ceil(total_read_size / data_chunk) * data_chunk
    print("ceiled_read_size", ceiled_read_size)
    usb_calls = int(int(ceiled_read_size / data_packs_size) / 128) * 128
    if usb_calls < 128:
        usb_calls = 128
    filename = str(obj.ss2_1b_cont_stream_DLP.ui.filename.text())
    file_path = "./usrsrc/ss2_1b_cont_stream_DLP/" + str(
        obj.ss2_1b_cont_stream_DLP.ui.file_path.text()
    )
    full_file_path = (
        str(pathlib.Path().absolute())
        + "/usrsrc/ss2_1b_cont_stream_DLP/"
        + str(obj.ss2_1b_cont_stream_DLP.ui.file_path.text())
    )
    data_per_frame = 2 * (row_end - row_start + 1) * column_en.count("1")
    print("data_per_frame", data_per_frame, "usb_calls", usb_calls)
    current_frame = 0
    count_frame = 0
    break_seq = int(obj.ss2_1b_cont_stream_DLP.ui.break_seq.isChecked())
    frame = np.zeros([256, 512], dtype=np.uint8)
    col = 0
    row = 0
    frames_per_bin = int(data_packs_size / read_size)
    col_no = column_en.count("1")
    col_en_int_v = [int(i) for i in column_en_bin]
    col_en_int_v = np.flip(col_en_int_v)
    check_bin_data = obj.ss2_1b_cont_stream_DLP.ui.check_mode.currentText()
    print(check_bin_data)
    if check_bin_data == "No check":
        check_headers = 0
        check_frame_headers = 0
    elif check_bin_data == "Headers only":
        check_headers = 1
        check_frame_headers = 0
        check_hist = 0
    elif check_bin_data == "Full check":
        check_headers = 1
        check_frame_headers = 1
        check_hist = 0
    elif check_bin_data == "Histogram lost frames":
        check_headers = 1
        check_frame_headers = 1
        check_hist = 1
    hist_list = {}
    last_frame_head = usb_call_init * frames_per_bin
    last_lost_frame_head = usb_call_init * frames_per_bin
    print(check_headers, check_frame_headers)
    if check_headers:
        for x_pkg in range(usb_call_init, usb_calls):
            f = open(
                file_path
                + "/"
                + str(x_pkg >> 11)
                + "/"
                + str(x_pkg >> 6)
                + "/"
                + filename
                + str(x_pkg)
                + ".bin",
                "rb",
            )
            bufferRd = bytearray(data_packs_size)
            bufferRd = f.read()
            f.close()
            data2 = np.frombuffer(bufferRd, dtype=np.uint32)
            if check_frame_headers:
                for x_col in range(0, (data_packs_size >> 2), data_per_frame):
                    data_32px = data2[x_col]
                    if check_hist == 0:
                        print(data_32px)
                    if ~((data_32px - last_frame_head) == 1):
                        if check_hist == 0:
                            print(
                                "acq bin file",
                                x_pkg,
                                "expected frame",
                                last_frame_head,
                                "frame",
                                data_32px,
                            )
                            return 0
                        elif data_32px < last_frame_head:
                            print(
                                "acq bin file",
                                x_pkg,
                                "expected frame",
                                usb_call_init * frames_per_bin,
                                "frame",
                                data2[0],
                            )
                            print("all frames checked")
                            return 0
                        else:
                            print(
                                "hist",
                                x_pkg,
                                data_32px,
                                data_32px - last_frame_head,
                                "         ",
                                last_frame_head - last_lost_frame_head,
                            )
                            last_lost_frame_head = data_32px
                    last_frame_head = data_32px
            else:
                data_32px = int(data2[0])
                print(data_32px, x_pkg)
                if (x_pkg * frames_per_bin) + 1 != data_32px:
                    print((x_pkg * frames_per_bin) + 1, data_32px)
                    print("bin file", x_pkg, "frame no", data_32px)
                    return 0
    print("for begins")
    for x_pkg in range(usb_call_init, usb_calls):
        f = open(
            file_path
            + "/"
            + str(x_pkg >> 11)
            + "/"
            + str(x_pkg >> 6)
            + "/"
            + filename
            + str(x_pkg)
            + ".bin",
            "rb",
        )
        bufferRd = bytearray(data_packs_size)
        bufferRd = f.read()
        f.close()
        data2 = np.frombuffer(bufferRd, dtype=np.uint32)
        for x_col in range(0, (data_packs_size >> 2)):
            data_32px = data2[x_col]
            if count_frame == 0:
                print("frane no", data_32px)
                break_seq = int(obj.ss2_1b_cont_stream_DLP.ui.break_seq.isChecked())
            if count_frame >= 0:
                col_loc = np.where(np.array(col_en_int_v) == 1)
                col_v = 15 - int(2 * col_loc[0][col >> 1] + (col & 1))
                # print(col_en_int_v,col_loc,col_v)
                for x_32 in range(0, 32):
                    col_32 = 31 - x_32
                    frame[255 - (row + row_start), 511 - (col_v * 32 + col_32)] = frame[
                        255 - (row + row_start), 511 - (col_v * 32 + col_32)
                    ] + ((data_32px >> x_32) & 1)
                col = col + 1
            if col == 2 * col_no:
                col = 0
                row = row + 1
                if row == (row_end - row_start + 1):
                    row = 0
            count_frame = count_frame + 1
            if count_frame == data_per_frame:
                count_frame = 0
                img = pg.ImageItem(np.transpose(frame))
                obj.ss2_1b_cont_stream_DLP.ui.graphicsView.addItem(img)
                pg.QtGui.QApplication.processEvents()
                if current_frame >= stop_frame:
                    return 0
                current_frame = current_frame + 1
                if ((current_frame & (1 << bits_size)) == ((1 << bits_size) - 1)) | (
                    bits_size == 0
                ):
                    frame = np.zeros([256, 512], dtype=np.uint8)
            if break_seq:
                return 0


def browse_patterns(obj):
    import PyQt5

    dir = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(
        None, "Choose Pattern Folder", "D:/Downloads/projector_frames"
    )
    obj.ss2_1b_cont_stream_DLP.ui.pattern_line.setText(dir)
