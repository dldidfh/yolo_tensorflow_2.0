import cv2

def check_input_image_and_boxes(args, train_data):
    for i in range(train_data.__len__() // args.batch_size):
        a = train_data.__getitem__(i)
        for j in range(args.batch_size):
            image = a[0][j]
            boxes = a[1][j]
            
            boxes = boxes[..., : 5 * args.box_per_grid]
            boxes = boxes[boxes != 0.]
            for x in range(len(boxes) // 5):
                box_x = boxes[0 + (x * 5)]
                box_y = boxes[1 + (x * 5)]
                box_w = boxes[2 + (x * 5)]
                box_h = boxes[3 + (x * 5)]
                box_xmin = int((box_x - box_w/2) * args.input_scale)
                box_ymin = int((box_y - box_h/2)* args.input_scale)
                box_xmax = int((box_x + box_w/2)* args.input_scale)
                box_ymax = int((box_y + box_h/2)* args.input_scale)
                image = cv2.rectangle(image,(box_xmin,box_ymin),(box_xmax,box_ymax),(244,244,0),1)

            cv2.imshow('asd', image )
            cv2.waitKey(0)