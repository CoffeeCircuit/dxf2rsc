"""
DXF2RSC

Software for converting aluminium DXF cross-sections to RSECTION files
Curent version tested on RSECTION v. 1.07.0006
"""

VERSION = "0.10.0"

import sys
import subprocess

# Check Python version
if not sys.version_info[0] == 3:
    print("Must be using Python 3!")
    input("Press Enter to exit...")
    sys.exit()


def logmessage(msg: str):
    log.insert("end", f"{msg}")
    log.see("end")


# Check all dependencies
_dependencies = ("numpy", "ezdxf", "suds-py3", "requests", "six", "mock", "xmltodict")
for _module in _dependencies:
    if _module not in sys.modules:
        subprocess.call(f"python -m pip install {_module} --user")


import numpy as np
import ezdxf
import tkinter as tk
import configparser
from os import listdir
from tkinter import ttk
from tkinter.filedialog import askdirectory, asksaveasfile, askopenfile
from math import sin, cos, pi, copysign
from threading import Thread


####### FILE IO  #######


def read_dxf(file: str):
    """
    Read contents of a single DXF file
    """
    try:
        return ezdxf.readfile(file)
    except IOError:
        print(f"No DXF file found.")
        sys.exit(1)
    except ezdxf.DXFStructureError:
        print(f"File {file} is invalid or corrupt")
        sys.exit(2)


####### HELPER FUNCTIONS FOR DXF ENTITY QUERY #######


def bulge_to_control_pt(
    start_p: list[float], end_p: list[float], bulge: float
) -> tuple[float]:
    """
    Converts the arc definition from using bulge to an intermediate control point.
    Positive bulge represents counterclockwise rotation of arc.
    >>> bulge_to_control_pt([394.2878, 158.5276], [371.9272, 151.2621], 0.324919696233)
    >>> (381.9271479735096, 158.52754967979382)
    """
    if len(start_p) == 2:
        start_p = np.array(start_p)
    else:
        raise ValueError("Expected List of length 2")
    if len(end_p) == 2:
        end_p = np.array(end_p)
    else:
        raise ValueError("Expected List of length 2")
    if isinstance(bulge, float):
        bulge = bulge
    else:
        raise TypeError("Bulge must be a float")
    chord = start_p - end_p
    r90 = np.array([[0, -1], [1, 0]]) * copysign(1, bulge)
    sagita = np.linalg.norm(chord) * 0.5 * abs(bulge)
    uchord = chord / np.linalg.norm(chord)
    control_p = (r90.dot(uchord)) * sagita + (start_p + end_p) * 0.5
    return tuple(control_p)


def get_all_arcs(msp):
    """
    Returns
    [[start_point.x, start_point.y, mid_point.x, mid_point.y, end_point.x, end_point.y],
     [start_point.x, start_point.y, mid_point.x, mid_point.y, end_point.x, end_point.y],

    """
    _arclist = []
    for e in msp.query("ARC"):
        cx, cy = (e.dxf.center.x, e.dxf.center.y)
        r = e.dxf.radius
        s_a = e.dxf.start_angle * pi / 180
        e_a = e.dxf.end_angle * pi / 180
        h_a = (e_a - s_a) / 2 + s_a  # in rad
        arc = (
            (cx + r * cos(s_a), cy + r * sin(s_a)),  # 1st point
            (cx + r * cos(h_a), cy + r * sin(h_a)),  # 2nd point
            (cx + r * cos(e_a), cy + r * sin(e_a)),  # 3rd point
        )
        _arclist.append(arc)
    return _arclist


def get_all_circles(msp):
    """
    Get all circle entities
    Returns:
    ((centre_x, center_y, radius),
     (centre_x, center_y, radius),
     ....
    )
    """
    return tuple(
        [(e.dxf.center.x, e.dxf.center.y, e.dxf.radius) for e in msp.query("CIRCLE")]
    )


def get_all_ellipses(msp):
    """
    Get all ellipse entities
    Returns:
    [[no1, centre_x, center_y, major_axis_x, major_axis_y, minor_axis_x, minor_axis_y],
     [no2, centre_x, center_y, major_axis_x, major_axis_y, minor_axis_x, minor_axis_y],
     ....
    ]
    """
    return tuple(
        [
            (
                e.dxf.center.x,
                e.dxf.center.y,
                e.dxf.major_axis.x,
                e.dxf.major_axis.y,
                e.minor_axis.x,
                e.minor_axis.y,
            )
            for e in msp.query("ELLIPSE")
        ]
    )


def get_all_points(msp):
    """
    Get all points coordinates from modelspace\n
    Returns:\n
    [[no1, x, y],
     [no2, x, y],
      ...
    ]
    """
    return tuple(
        [
            (
                e.dxf.location.x,
                e.dxf.location.y,
            )
            for e in msp.query("POINT")
        ]
    )


def get_all_lines(msp):
    """
    Get all lines from modelspace\n
    Returns:\n
    ((start_x, start_y, end_x, end_y),
     (start_x, start_y, end_x, end_y),
      ...
    )
    """
    return tuple(
        [
            (
                e.dxf.start.x,
                e.dxf.start.y,
                e.dxf.end.x,
                e.dxf.end.y,
            )
            for e in msp.query("LINE")
        ]
    )


def get_all_polylines(msp):
    """
    Get all polylines from modelspace\n
    Returns:\n
    ((x, y, bulge),
     (x, y, bulge),
      ...
    )

    """
    return tuple([tuple(e.get_points("xyb")) for e in msp.query("LWPOLYLINE")])


#### RAY CASTING FUNCTIONS ####


def ray_intersecting_segment(
    hray: float, line: tuple[tuple[float]], xmin: float, xmax: float
) -> None | tuple[float]:
    """
    Check if horizontal ray is intersecting a line segment.\n
    If get_point = True -> Returns intersection point coordinates
    """
    x1, y1 = xmin, hray
    x2, y2 = xmax, hray
    x3, y3 = line[0]
    x4, y4 = line[1]
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return None
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return None
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)


def ray_intersecting_circle(
    hray: float,
    centre: list[float],
    radius: float,
) -> bool:
    """
    Checks if horizontal ray is intersecting (is tangent to) a circle of given centre and radius.\n
    """
    x_c, y_c = centre
    (x1, y1), (x2, y2) = (x_c - radius - 1, hray), (x_c + radius + 1, hray)
    return (
        abs((y2 - y1) * x_c + (x1 - x2) * y_c + y1 * (x2 - x1) - x1 * (y2 - y1))
        / ((y2 - y1) ** 2 + (x1 - x2) ** 2) ** 0.5
        <= radius
    )


def dist_p_to_line(pt: list[float], line: list[float]):
    """
    Distance from point to line.\n
    """
    x, y = pt
    (x1, y1), (x2, y2) = line
    # line general form coeficients
    # A = y2 - y1
    # B = x1 - x2
    # C = y1 * (x2 - x1) - x1 * (y2 - y1)
    return (
        abs((y2 - y1) * x + (x1 - x2) * y + y1 * (x2 - x1) - x1 * (y2 - y1))
        / ((y2 - y1) ** 2 + (x1 - x2) ** 2) ** 0.5
    )


def yrange(y_min, y_max, steps):
    _list = []
    _yrange = range(steps + 1)
    f = (y_max - y_min) / steps
    for i in _yrange:
        _list.append(y_min + i * f)
    return _list


def gensection(_file: str):
    from RSECTION.initModel import Calculate_all, Model, closeModel, session
    from RSECTION.BasicObjects.material import Material
    from RSECTION.BasicObjects.point import Point
    from RSECTION.BasicObjects.line import Line
    from RSECTION.BasicObjects.part import Part
    from RSECTION.BasicObjects.opening import Opening
    from RSECTION.BasicObjects.element import Element

    logmessage("RUNNING: Connecting to server...")
    global PATH
    doc = read_dxf(f"{PATH}" + "\\" + f"{_file}")
    logmessage(f"RUNNING: Reading file {_file}")
    msp = doc.modelspace()
    poly_list = get_all_polylines(msp)
    Model(True, f"{_file}", delete_all=True)
    Material(1, "S235")
    NODE_ID = 0
    LINE_ID = 0
    PARTID = 0
    OPENINGID = 0
    POINTSLIST = dict()
    LINESLIST = list()

    ###### START OF POLYLINE SEGMENT ######
    for poly in poly_list:
        require_ctrl_pt = False
        key_1 = ""
        _tpdict = {}
        _tldict = {}
        _sp, _ep, _b = (), (), 0.0
        for j, vertex in enumerate(poly):
            x, y, bulge = vertex
            if j >= 1 and j <= len(poly) - 1:
                NODE_ID += 1
                LINE_ID += 1
                Point(NODE_ID, x / 1000, y / 1000)
                _tpdict[str(NODE_ID)] = (x / 1000, y / 1000)

                # check if previous point had a bulge
                if require_ctrl_pt == True:
                    _ep = (x / 1000, y / 1000)
                    ctrl_pt = bulge_to_control_pt(_sp, _ep, _b)
                    Line.Arc(
                        LINE_ID,
                        [NODE_ID - 1, NODE_ID],
                        [ctrl_pt[0], ctrl_pt[1]],
                    )
                    _tldict[str(LINE_ID)] = (f"{NODE_ID - 1}, {NODE_ID}", "A")
                else:
                    Line(LINE_ID, f"{NODE_ID-1} {NODE_ID}")
                    _tldict[str(LINE_ID)] = (f"{NODE_ID - 1}, {NODE_ID}", "L")

                # set flag for curent point if it has a bulge
                if bulge != 0:
                    require_ctrl_pt = True
                    _sp = (x / 1000, y / 1000)
                    _b = bulge
                else:
                    require_ctrl_pt = False
                    _sp = ()
                    _ep = ()
                    _b = 0.0

            else:
                NODE_ID += 1
                Point(NODE_ID, x / 1000, y / 1000)
                _tpdict[str(NODE_ID)] = (x / 1000, y / 1000)

                if bulge != 0:
                    require_ctrl_pt = True
                    _sp = (x / 1000, y / 1000)
                    _b = bulge

            if j == len(poly) - 1:
                LINE_ID += 1
                key_1 = next(iter(_tpdict))
                if require_ctrl_pt == True:
                    _ep = _tpdict[key_1]
                    ctrl_pt = bulge_to_control_pt(_sp, _ep, _b)
                    Line.Arc(
                        LINE_ID,
                        [key_1, NODE_ID],
                        [ctrl_pt[0], ctrl_pt[1]],
                    )
                    _tldict[str(LINE_ID)] = (f"{NODE_ID}, {key_1}", "A")
                else:
                    Line(LINE_ID, f"{NODE_ID} {key_1}")
                    _tldict[str(LINE_ID)] = (f"{NODE_ID}, {key_1}", "L")
                POINTSLIST.update(_tpdict)
        LINESLIST.append(_tldict)
    del (
        _b,
        _ep,
        _sp,
        _tldict,
        _tpdict,
        j,
        bulge,
        require_ctrl_pt,
        vertex,
    )
    ###### END OF POLYLINE SEGMENT ######

    ###### START OF PART/OPENING SEGMENT ######
    if len(LINESLIST) == 1:
        key_n = [*LINESLIST[0].keys()][-1]
        PARTID += 1
        Part(PARTID, f"{key_1}-{key_n}")
    else:
        # get bounding box
        _Xs = [x for x, _ in POINTSLIST.values()]
        _Ys = [y for _, y in POINTSLIST.values()]
        xmin = min(_Xs) - 0.005
        xmax = max(_Xs) + 0.005
        ymin = min(_Ys)
        ymax = max(_Ys)
        del _Xs, _Ys

        # do raycasting
        rayrange = [(ymax - ymin) / 20 * i + ymin for i in range(20)]
        ray_intersections = []
        for hray in rayrange:
            line_intersections = []
            for polygon in LINESLIST:
                for line_no, line_def in polygon.items():
                    line_end_coords = tuple(
                        POINTSLIST[point.strip()] for point in line_def[0].split(",")
                    )
                    line_type = line_def[-1]
                    _ipt = ray_intersecting_segment(hray, line_end_coords, xmin, xmax)
                    if _ipt:
                        line_intersections.append((line_no, _ipt[0]))
                _t = tuple(
                    tup[0]
                    for tup in list(sorted(line_intersections, key=lambda x: x[1]))
                )
            ray_intersections.append(_t)

        del (
            line_no,
            line_def,
            line_type,
            line_end_coords,
            line_intersections,
            polygon,
            hray,
            rayrange,
            _ipt,
            x,
            y,
            key_1,
            xmax,
            xmin,
            ymax,
            ymin,
        )

        # do part generation
        is_part = []
        old_rintersections = ()
        for _rintersections in ray_intersections:
            if old_rintersections == _rintersections:
                continue
            else:
                if len(_rintersections):
                    (
                        is_part.append(_rintersections[0])
                        if _rintersections[0] not in is_part
                        else None
                    )
                    (
                        is_part.append(_rintersections[-1])
                        if _rintersections[-1] not in is_part
                        else None
                    )
            old_rintersections = _rintersections

        for polygon in LINESLIST:
            line_nos = set(polygon.keys())
            if line_nos.issuperset(set(is_part)):
                PARTID += 1
                polygon.keys()
                Part(PARTID, f"{list(polygon.keys())[0]}-{list(polygon.keys())[-1]}")
            else:
                OPENINGID += 1
                Opening(
                    OPENINGID, f"{list(polygon.keys())[0]}-{list(polygon.keys())[-1]}"
                )
    ###### END OF PART/OPENING SEGMENT ######

    logmessage(f"RUNNING: Calculating section... ")
    Calculate_all()

    logmessage(f"RUNNING: Saving model...")
    Model.clientModel.service.save(f"{PATH}" + "\\" + f"{_file}")
    closeModel(f"{_file.split('.')[0]}", save_changes=True)
    session.close()
    return 0


##### GUI FUNCTIONS #####


def close(event):
    master.quit()


def saveconfigfile():
    """
    Save the current state to a configuration file.
    """
    global VERSION
    config = configparser.ConfigParser()
    if lstFiles.size():
        config["DXF2RSC"] = {"VERSION": VERSION}
        config["FOLDER"] = {"PATH": txtPath.get("1.0", "end")}
        config.add_section("FILES")
        for i in range(lstFiles.size()):
            config.set("FILES", str(i + 1), lstFiles.get(i))
        files = [("Config Files", "*.ini")]
        file = asksaveasfile(filetypes=files, defaultextension=files)
        with open(file.name, "w") as fp:
            config.write(fp)
            logmessage(f"Configuration file written to {file.name}")
    else:
        logmessage("No file loaded")


def openconfigfile():
    global CONFIGFILE, VERSION
    config = configparser.ConfigParser()
    files = [("Config Files", "*.ini")]
    CONFIGFILE = askopenfile(filetypes=files, defaultextension=files)
    config.read(CONFIGFILE.name)
    if config.get("DXF2RSC", "VERSION") == VERSION:
        logmessage("----TODO----")  # TODO write the file parser
    else:
        logmessage("Versions do not match.")


def getpath():
    global PATH
    txtPath.delete("1.0", "end")
    lstFiles.delete("0", "end")
    _path = askdirectory()
    PATH = _path
    txtPath.insert("end", _path)


def setcontent(event):
    # return txtPath.get('1.0','end')
    if txtPath.edit_modified():
        _path = txtPath.get("1.0", "end")
        for i, file in enumerate(listdir(_path.rstrip())):
            lstFiles.insert(i + 1, f"{file}") if file.endswith(".dxf") else None


def clearpath():
    global PATH
    PATH = ""
    txtPath.delete("1.0", "end")
    lstFiles.delete("0", "end")
    txtPath.edit_modified(False)


def eventF5(event):  # run_sel
    run_sel_threaded()


def eventShiftF5(event):  # run_all
    run_all_threaded()


def run_sel():
    if not lstFiles.curselection():
        logmessage("Select file from list")
    else:
        progress.start()
        _selection = lstFiles.get(lstFiles.curselection())
        gensection(_selection)
        logmessage(f"DONE")
        progress.stop()


def run_all():
    global PATH
    # check if files exist
    if not lstFiles.size():
        logmessage("No DXF files found")

    else:
        no_files = lstFiles.size()
        prgbar_step = 100 / (no_files)
        for i in range(no_files):  # replace 2 with lstFiles.size()
            gensection(lstFiles.get(i))
            progress.step(prgbar_step)
        logmessage(f"DONE")


def run_sel_threaded():
    thr = Thread(target=run_sel)
    thr.start()


def run_all_threaded():
    thr = Thread(target=run_all)
    thr.start()


######### MAIN BLOCK ############
if __name__ == "__main__":

    # Run GUI
    master = tk.Tk()
    master.title("DXF to RSection File conversion")
    master.geometry("798x600")  # 798x600
    master.resizable(0, 0)

    menu = tk.Menu(master)

    master.config(menu=menu)
    filemenu = tk.Menu(menu)
    menu.add_cascade(label="File", menu=filemenu)
    filemenu.add_command(label="Save configuration", command=saveconfigfile)
    filemenu.add_command(label="Open configuration", command=openconfigfile)
    filemenu.add_separator()
    filemenu.add_command(label="Quit (F10)", command=master.quit)
    helpmenu = tk.Menu(menu)
    menu.add_cascade(label="Help", menu=helpmenu)
    helpmenu.add_command(label="Usage")
    helpmenu.add_command(label="About")

    btnAdd = tk.Button(master, text="Add Folder", width=15, height=1, command=getpath)
    btnAdd.grid(row=0, column=0)

    btnClear = tk.Button(master, text="Clear", width=15, height=1, command=clearpath)
    btnClear.grid(row=0, column=1)

    txtPath = tk.Text(master, height=1, width=70)
    txtPath.config(wrap="none")
    txtPath.grid(row=0, column=2)
    txtPath.bind("<<Modified>>", setcontent)

    lstFiles = tk.Listbox(master, width=132, height=30, justify="left")
    lstFiles.grid(row=1, columnspan=3)

    log = tk.Listbox(
        master, width=132, background="lightgray", height=4, justify="left"
    )
    log.grid(row=3, columnspan=3)

    btnRunSel = tk.Button(
        master, text="Run selected (F5)", width=15, command=run_sel_threaded
    )
    btnRunSel.grid(row=2, column=0)
    master.bind("<F5>", eventF5)
    master.bind("<F10>", close)

    btnRunAll = tk.Button(
        master, text="Run all (Shit+F5)", width=15, command=run_all_threaded
    )
    btnRunAll.grid(row=2, column=1)
    master.bind("<Shift-F5>", eventShiftF5)

    progress = ttk.Progressbar(
        master, orient="horizontal", length=560, mode="indeterminate"
    )
    progress.grid(row=2, column=2)

    master.mainloop()
