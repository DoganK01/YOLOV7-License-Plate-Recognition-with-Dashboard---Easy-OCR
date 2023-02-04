import math
import easyocr
import cv2

reader = easyocr.Reader(['en'],gpu=True)
class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, label,image = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            x1 , y1, w1, h1 = int(x), int(y), int(w), int(h)
            # crop the image to bounding box and pass it to readtext
            tempIMG = image[y1:h1,x1:w1]
            gray = cv2.cvtColor(tempIMG,cv2.COLOR_RGB2GRAY)
            result = reader.readtext(gray)
            text = ""
            # save the text
            for res in result:
                if len(result) >= 1 and len(res[1]) > 6 and res[2] > 0.2:
                    text = res[1]

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id,label,str(text)])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count,label,str(text)])
                self.id_count += 1


        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _,_ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



