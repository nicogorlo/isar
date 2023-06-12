import cv2
import numpy as np
import os
import pickle
import json
from pathlib import Path

import torch

class HabitatSceneDataReader():
    def __init__(self, folder_raw_frames, out_folder, scene_name) -> None:
        self.data_path = folder_raw_frames
        self.scene_name = scene_name
        self.rgb_path = os.path.join(self.data_path, self.scene_name, "color")
        self.semantic_path = os.path.join(self.data_path, self.scene_name, "semantic_raw")
        self.out_dir = os.path.join(out_folder, self.scene_name)
        os.makedirs(os.path.join(self.out_dir, 'rgb/'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'semantics/'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'embeddings/'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'combined_vis/'), exist_ok=True)
        self.count = 0

        self.initial_coordinates = None

        cv2.namedWindow("RGB")
        cv2.namedWindow("Semantics")
        cv2.setMouseCallback("RGB",self.select)

    def load_first_img(self):
        self.image_names = [n.split('.')[0] for n in sorted(os.listdir(self.rgb_path))]
        img0 = cv2.imread(os.path.join(self.rgb_path, self.image_names[0] + ".jpg"))
        all_semantic_annotations = np.load(os.path.join(self.semantic_path, self.image_names[0] + ".npy"))

        return img0, all_semantic_annotations

    def __call__(self):

        img0, all_semantic_annotations = self.load_first_img()

        cv2.imshow("RGB", img0)

        print("select object to track, press ENTER to confirm, press ESC to quit")
        while True:
            key = cv2.waitKey(0)
            if key == 27: # ESC
                cv2.destroyAllWindows()
                exit()
            if key == 13: # ENTER
                if self.initial_coordinates is None:
                    print ("please select an object to track before pressing ENTER")
                    continue
                else:
                    print("confirmed")
                    print("initial coordinates: ", self.initial_coordinates)
                    break

        
        self.selected_class_id = all_semantic_annotations[self.initial_coordinates[1], self.initial_coordinates[0]]
        print("selected class ID: ", self.selected_class_id)
        self.selected_class_mask = (all_semantic_annotations == self.selected_class_id)

        cv2.imshow("Semantics", self.show_mask(self.selected_class_mask, random_color=False))
        cv2.waitKey(int(1000/30))

    def select(self, event, x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            self.initial_coordinates = (x,y)

    def show_mask(self, mask: np.ndarray, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([0, 0, 255, 1.0])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        return mask_image

    def get_mask(self, image):
        img = cv2.imread(os.path.join(self.rgb_path, image + ".jpg"))
        semantic_annot = np.load(os.path.join(self.semantic_path, image + ".npy"))

        mask = (semantic_annot == self.selected_class_id)

        show_mask = self.show_mask(mask, random_color=False)

        return img, show_mask
    
    def visualize_and_save(self, img, show_mask):
        combined = cv2.addWeighted(img.astype("uint8"), 0.5, show_mask[:, :, :3].astype("uint8"), 0.5, 0)
        cv2.imshow("RGB", combined)
        cv2.imwrite(os.path.join(self.out_dir, "rgb/" , str(self.count).zfill(7) + ".jpg"), img)
        cv2.imshow("Semantics", show_mask)
        cv2.imwrite(os.path.join(self.out_dir, "semantics/" , str(self.count).zfill(7) + ".jpg"), show_mask)
        cv2.imwrite(os.path.join(self.out_dir, "combined_vis/" , str(self.count).zfill(7) + ".jpg"), combined)

        
        cv2.waitKey(int(1000/60))
    
    def get_prompt(self):
        return self.initial_coordinates
    
class DavisSceneDataReader(HabitatSceneDataReader):
    def __init__(self, folder_raw_frames, out_folder, semantic_folder, scene_name):
        super().__init__(folder_raw_frames, out_folder, scene_name)
        self.semantic_path = os.path.join(semantic_folder, self.scene_name)
        self.rgb_path = os.path.join(self.data_path, self.scene_name)
        self.image_names = [n.split('.')[0] for n in sorted(os.listdir(self.rgb_path))]

    def load_first_img(self):
        img0 = cv2.imread(os.path.join(self.rgb_path, self.image_names[0] + ".jpg"))
        all_semantic_annotations = cv2.imread(os.path.join(self.semantic_path, self.image_names[0] + ".png"), cv2.IMREAD_GRAYSCALE)
        return img0, all_semantic_annotations
    
    def get_mask(self, image):
        img = cv2.imread(os.path.join(self.rgb_path, image + ".jpg"))
        semantic_annot = cv2.imread(os.path.join(self.semantic_path, image + ".png"), cv2.IMREAD_GRAYSCALE)

        mask = (semantic_annot == self.selected_class_id)

        show_mask = self.show_mask(mask, random_color=False)

        return img, show_mask
          

def habitat_dataset_reader(folder_habitat_raw_frames, out_folder):

    prompt_dict = {}
    for scene in sorted(os.listdir(folder_habitat_raw_frames)):
        print(scene)
        dr = HabitatSceneDataReader(folder_habitat_raw_frames, out_folder, scene)
        dr()
        x, y = dr.get_prompt()
        scene_prompt_dict = {'x': x, 'y': y}
        prompt_dict[scene] = scene_prompt_dict
        print(prompt_dict)

        for image in sorted(dr.image_names):

            img, show_mask = dr.get_mask(image)
            dr.visualize_and_save(img, show_mask)

            dr.count +=1


    prompt_dict_path = os.path.join(out_folder, "prompt_dict.json")
    Path(prompt_dict_path).touch(exist_ok=True)
    with open(prompt_dict_path, 'w') as f:
        json.dump(prompt_dict, f)

def davis_dataset_reader(folder_davis_raw_frames, out_folder, semantic_folder, embedding_folder):

    prompt_dict = {}
    for scene in sorted(os.listdir(folder_davis_raw_frames)):
        print(scene)
        dr = DavisSceneDataReader(folder_davis_raw_frames, out_folder, semantic_folder, scene)
        dr()
        x, y = dr.get_prompt()
        scene_prompt_dict = {'x': x, 'y': y}
        prompt_dict[scene] = scene_prompt_dict
        print(prompt_dict)

        for image in sorted(dr.image_names):

            img, show_mask = dr.get_mask(image)
            dr.visualize_and_save(img, show_mask)

            dr.count+=1

    prompt_dict_path = os.path.join(out_folder, "prompt_dict.json")
    Path(prompt_dict_path).touch(exist_ok=True)
    with open(prompt_dict_path, 'w') as f:
        json.dump(prompt_dict, f)

def main():

    Dataset = input("Which dataset do you want to use? (Habitat or Davis) ")
    if Dataset == 'Habitat':
        habitat_dataset_reader(folder_habitat_raw_frames = "", 
                            out_folder = "")
    if Dataset == 'Davis':
        davis_dataset_reader(folder_davis_raw_frames = "",
                            out_folder = "",
                            semantic_folder = "",
                            embedding_folder= "")


    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()