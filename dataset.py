import numpy as np 
import tensorflow as tf 
import cv2
import os 
import albumentations as Album
import random


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, args, mode=0) -> None:
        super().__init__()
        self.args = args
        self.mode = mode 
        self.batch_size = self.args.batch_size
        self.output_shape = self.args.box_per_grid * 5 + self.args.class_num
        # 학습시와 테스트시 데이터셋을 다르게 하기 위해 
        if self.mode == 0:
            dir_path = self.args.train_dir_path 
        else:
            dir_path = self.args.test_dir_path 
        
        # # 클레스 이름과 값을 가져옴 
        # self.class_names_dict = {}
        # with open(self.args.class_name_file) as f:
        #     self.class_names = f.read().splitlines()
        #     for i in range(len(self.class_names)):
        #         self.class_names_dict[self.class_names[i]] = i
        # 데이터의 경로를 저장할 리스트 
        self.txt_path_list = []
        self.img_path_list = []
        # 설정된 경로에서 모든 파일을 읽음 
        for file_name in os.listdir(dir_path):
            root, child = os.path.splitext(file_name)
            if child == '.txt':
                validation_state, ext = self.check_img_and_txt(dir_path, root)
                if validation_state:
                    self.txt_path_list.append(os.path.join(dir_path, root + '.txt')) # file_name도 같지만 보기 쉽게 .txt로 설정 
                    self.img_path_list.append(os.path.join(dir_path, root +   ext))
        self.boxes_and_labels = []
        for i, txt_path in enumerate(self.txt_path_list):
            self.boxes_and_labels.append(self.parse_txt(txt_path, self.img_path_list[i]))
        self.data_index = np.zeros([len(self.img_path_list)], np.int32)
        for i in range(len(self.data_index)):
            self.data_index[i] = i
        np.random.shuffle(self.data_index)



    def __len__(self):
        return len(self.img_path_list) // self.batch_size
    def get_size(self):
        return len(self.img_path_list)
    def on_epoch_end(self):
        np.random.shuffle(self.data_index)


    def __getitem__(self, idx):
        # 배치사이즈 만큼 스플릿 
        train_image = []
        # 학습데이터의 정답으로 들어가니 모델의 아웃풋과 쉐입이 똑같아야함 
        train_label = np.zeros([self.batch_size,self.args.grid_size,self.args.grid_size,self.output_shape], dtype=np.float32)

        for index, file_index in enumerate(self.data_index[idx*self.batch_size : (idx+1)*self.batch_size]):
             # img_path_list = 이미지 경로, 
             # boxes_and_labels = [ center x, center y, width, height, class_num]
            image = self.img_path_list[file_index]
            bboxes = self.boxes_and_labels[file_index]
            image, bboxes = self.image_augmentations(image, bboxes)
            train_image.append(image / 255.)
            for box in bboxes:
                # 그리드 위치에 맞게 
                grid_coord = [self.args.grid_size * box[0], self.args.grid_size * box[1]]
                grid_i = int(grid_coord[1])
                grid_j = int(grid_coord[0])
                for i in range(self.args.box_per_grid):
                # 만약 해당 그리드에 객체가 없으면 - 하나의 그리드에 최대 3개 - args.box_per_grid 만큼만 가질 수 있다 
                    if train_label[index,grid_i, grid_j, 5*i:5*(i+1)].all() == 0.:
                        # 클래스 값을 1 로 바꿈 - 원핫 ? 
                        train_label[index,grid_i, grid_j,5*self.args.box_per_grid + int(box[4])] = 1 
                        # 박스 x, y , w, h, confidence 값을 넣어준다 
                        # print([box[0], box[1], box[2], box[3], 1])
                        train_label[index,grid_i, grid_j,5*i:5*(i+1)] = [box[0], box[1], box[2], box[3], 1]
                        break

        # # 배치만 큼 돌려서 배치 사이즈 만큼 묶음 
        # for i in range(len(batch_image)):
        #     img_path = batch_image[i]
        #     image = cv2.imread(img_path)
        #     image = cv2.resize(image, (self.args.input_scale, self.args.input_scale))
        #     image = image / 255.
        #     train_image.append(image)


        return np.array(train_image), np.array(train_label)

    def image_augmentations(self, image, bboxes):
        image = cv2.imread(image)
        H,W = image.shape[:2]
        transform = Album.Compose([
            Album.OneOf([
                Album.HorizontalFlip(p=0.5),
                Album.Blur(p=0.5),
                Album.RandomCrop(random.randint(int(H*0.3), H),random.randint(int(W*0.3), W)),
                Album.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.5),
            ]),
            
            # 리사이즈 - 인풋스케일로 
            Album.LongestMaxSize(max_size=self.args.input_scale),
            # 사진 비율 똑같게 하기위해서 남는 공간에 패딩 추가 
            Album.PadIfNeeded(min_height=self.args.input_scale, min_width=self.args.input_scale, mask_value=0,p=1,border_mode=cv2.BORDER_CONSTANT),
        ], bbox_params=Album.BboxParams(format='yolo', min_visibility=0.5))

        
        transformed = transform(image=image, bboxes=bboxes)
        trans_image = transformed['image']
        trans_bboxes = transformed['bboxes']    

        return trans_image, trans_bboxes


    def parse_txt(self, file_path, img_path):
        h,w = cv2.imread(img_path).shape[:2]
        with open(file_path, 'r') as rd:
            boxes = rd.readlines()
            # output_label = np.zeros([self.args.grid_size,self.args.grid_size,self.output_shape], dtype=np.float32)
            output_label = []
            for box in boxes:
                box = box.strip().split()
                class_num = box[0]
                center_x =  float(box[1]) #* w / self.args.input_scale
                center_y = float(box[2])# * h / self.args.input_scale
                box_width = float(box[3]) #* w / self.args.input_scale
                box_height = float(box[4]) #* h / self.args.input_scale
                output_label.append([center_x, center_y, box_width, box_height, class_num])

               
        return output_label
    # def parse_txt(self, file_path, img_path):
    #     h,w = cv2.imread(img_path).shape[:2]
    #     with open(file_path, 'r') as rd:
    #         boxes = rd.readlines()
    #         output_label = np.zeros([self.args.grid_size,self.args.grid_size,self.output_shape], dtype=np.float32)

    #         for box in boxes:
    #             box = box.strip().split()
    #             class_num = int(box[0])
    #             center_x =  float(box[1]) #* w / self.args.input_scale
    #             center_y = float(box[2])# * h / self.args.input_scale
    #             box_width = float(box[3]) #* w / self.args.input_scale
    #             box_height = float(box[4]) #* h / self.args.input_scale


    #             grid_coord = [self.args.grid_size * center_x, self.args.grid_size * center_y]
    #             grid_i = int(grid_coord[1])
    #             grid_j = int(grid_coord[0])
    #             for i in range(self.args.box_per_grid):
    #             # 만약 해당 그리드에 객체가 없으면 - 하나의 그리드에 최대 3개 - args.box_per_grid 만큼만 가질 수 있다 
    #                 if output_label[grid_i, grid_j, 5*i:5*(i+1)].all() == 0.:
    #                     # 클래스 값을 1 로 바꿈 - 원핫 ? 
    #                     output_label[grid_i, grid_j,5*self.args.box_per_grid + class_num] = 1 
    #                     # 박스 x, y , w, h, confidence 값을 넣어준다 
    #                     # print([center_x, center_y, box_width, box_height, 1])
    #                     output_label[grid_i, grid_j,5*i:5*(i+1)] = [center_x, center_y, box_width, box_height, 1]
    #                     break

    #     return output_label
    
    def check_img_and_txt(self, root_dir, root):
        extention_list = ['.jpg', 'png', '.bmp', '.JPG']
        # if os.path.exists(os.path.join(root_dir,root + '.txt')):
        try:
            for ext in extention_list:
                if os.path.exists(os.path.join(root_dir, root + ext)):
                    return True, ext
        except:
            print('txt dont have image file : \t {}'.format(os.path.join(root_dir, root + '.txt')))
            return False, '...'