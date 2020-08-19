import json
import os
import cv2

''''
Images taken from Berkley Deep Drive Dataset.
For more info about this dataset:
BDD github repo: https://github.com/ucbdrive/bdd100k
                BDD JSON FILE FORMAT:        
- name: string
- url: string
- videoName: string (optional)
- attributes:
    - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
    - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
    - timeofday: "daytime|night|dawn/dusk|undefined"
- intrinsics
    - focal: [x, y]
    - center: [x, y]
    - nearClip:
- extrinsics
    - location
    - rotation
- timestamp: int64 (epoch time ms)
- frameIndex: int (optional, frame index in this video)
- labels [ ]:
    - id: int32
    - category: string (classification)
    - manualShape: boolean (whether the shape of the label is created or modified manually)
    - manualAttributes: boolean (whether the attribute of the label is created or modified manually)
    - score: float (the confidence or some other ways of measuring the quality of the label.)
    - attributes:
        - occluded: boolean
        - truncated: boolean
        - trafficLightColor: "red|green|yellow|none"
        - areaType: "direct | alternative" (for driving area)
        - laneDirection: "parallel|vertical" (for lanes)
        - laneStyle: "solid | dashed" (for lanes)
        - laneTypes: (for lanes)
    - box2d:
       - x1: float
       - y1: float
       - x2: float
       - y2: float
   - box3d:
       - alpha: (observation angle if there is a 2D view)
       - orientation: (3D orientation of the bounding box, used for 3D point cloud annotation)
       - location: (3D point, x, y, z, center of the box)
       - dimension: (3D point, height, width, length)
   - poly2d: an array of objects, with the structure
       - vertices: [][]float (list of 2-tuples [x, y])
       - types: string (each character corresponds to the type of the vertex with the same index in vertices. ‘L’ for vertex and ‘C’ for control point of a bezier curve.
       - closed: boolean (closed for polygon and otherwise for path)

'''

weatherAtt2keep = ['clear', 'overcast', 'undefined', 'partly', 'cloudy']
sceneAtt2keep = ['tunnel', 'highway']
timeAtt2keep = ['daytime', 'dawn/dusk', 'undefined']


def filterData(weatherAtt2keep, sceneAtt2keep, timeAtt2keep, labelsJson):
    ''''
    Filtering out the unwanted images containing unwanted attributes
    Inputs: which attributes TO KEEP regarding weather, scene and time-of-day
            Json containing BDD info, labels and attributes
    Outputs: data dictionary containing wanted images only (after filter applied)

    All wanted images should be relocated to 'used_images' folder under 'bdd100k' using the outputed dictionary
    '''
    data_dict = {}
    with open(labelsJson) as json_file:
        data = json.load(json_file)                 # data is a list of dictionaries

    for example in data:
        if example['attributes']['weather'] in weatherAtt2keep and \
                example['attributes']['scene'] in sceneAtt2keep and \
                example['attributes']['timeofday'] in timeAtt2keep:
            data_dict[example['name']] = example
    return data_dict


def relocate_wanted_images(data, orig_im_dir, new_im_dir):
    ''''
    Relocating the wanted data (after filtering) to different directory
    Inputs: data -- dictionary { KEY - image name, VALUE - image data }
    '''
    for img_name in data:
        os.rename(os.path.join(orig_im_dir,img_name), os.path.join(new_im_dir,img_name))
    return


def preview_filtered_data(data, im_dir):
    ''''
    Sanity check, preview data to verify data properly filtered
    INPUTS: data -- dictionary { KEY - image name, VALUE - image data }
    HIT 0 TO SEE ANOTHER IMAGE, CONSEQUENTLY
    '''
    for img_name in data:
        img = cv2.imread(os.path.join(im_dir, img_name))
        cv2.imshow('Example Image', img)
        stop = int(chr(cv2.waitKey()))
        cv2.destroyAllWindows()
        if stop != 0:
            break
    return







