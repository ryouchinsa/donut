# donut

Using the script [general_json2yolo.py](https://github.com/ryouchinsa/Rectlabel-support/blob/master/general_json2yolo.py), you can convert the RLE mask with holes to the YOLO segmentation format.

The RLE mask is converted to a parent polygon and a child polygon using `cv2.findContours()`.
The parent polygon points are sorted in clockwise order. 
The child polygon points are sorted in counterclockwise order.
Detect the nearest point in the parent polygon and in the child polygon.
Connect those 2 points with narrow 2 lines.
So that the polygon with a hole is saved in the YOLO segmentation format.

```
def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1]);
    return value < 0

def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2);
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour

def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)

def mask2polygon(image):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons 

def rle2polygon(segmentation):
    if isinstance(segmentation["counts"], list):
        segmentation = mask.frPyObjects(segmentation, *segmentation["size"])
    m = mask.decode(segmentation) 
    m[m > 0] = 255
    polygons = mask2polygon(m)
    return polygons
```

The RLE mask.

![スクリーンショット 2023-11-22 1 57 52](https://github.com/ryouchinsa/Rectlabel-support/assets/1954306/df2dfad5-a13f-49c6-818a-8aba1d3ce83c)

The converted YOLO segmentation format.

![スクリーンショット 2023-11-22 2 11 14](https://github.com/ryouchinsa/Rectlabel-support/assets/1954306/cc34027a-075f-4b67-8de7-bd2dddddac59)

To run the script, put the COCO JSON file coco_train.json into `datasets/coco/annotations`.
Run the script. `python general_json2yolo.py  `
The converted YOLO txt files are saved in `new_dir/labels/coco_train`.

![スクリーンショット 2023-11-23 16 39 21](https://github.com/ultralytics/JSON2YOLO/assets/1954306/c3d98120-66f5-4cb8-b74d-a500f0bd811d)

Edit use_segments and use_keypoints in the script.

```
if __name__ == '__main__':
    source = 'COCO'

    if source == 'COCO':
        convert_coco_json('../datasets/coco/annotations',  # directory with *.json
                          use_segments=True,
                          use_keypoints=False,
                          cls91to80=False)
```

To convert the COCO bbox format to YOLO bbox format.
```
use_segments=False,
use_keypoints=False,
```

To convert the COCO segmentation format to YOLO segmentation format.
```
use_segments=True,
use_keypoints=False,
```

To convert the COCO keypoints format to YOLO keypoints format.
```
use_segments=False,
use_keypoints=True,
```

This script originates from Ultralytics [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) repository.
We hope this script would help your business.
