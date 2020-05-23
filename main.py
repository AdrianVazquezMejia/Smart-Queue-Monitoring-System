#%%writefile person_detect.py
#python3 main.py --model /home/adrian-estelio/Documents/vision/PeopleCounterApp/model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph --video /home/adrian-estelio/Documents/vision/PeopleCounterApp/resources/Pedestrian_Detect_2_1_1.mp4

import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import numpy as np

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        Load the model to the inference Engine
        '''
        self.core = IECore()
        self.exec_net = self.core.load_network(self.model,self.device)
        print("Model loaded Successfully")
        #Implement supported layers
        return
        
    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        frame = self.preprocess_input(image)
        self.exec_net.infer({self.input_blob:frame})
        print("Inference ran successfully")
        self.output_blob = next(iter(self.model.outputs))
        output = self.exec_net.requests[0].outputs[self.output_blob]
        print("Output got successfully")
        coords = self.preprocess_outputs(output)
        print("Coords got successfully")
        out_frame = self.draw_outputs(coords,image)
        print("Bounding Box drew successfully")
        return coords,out_frame
    
    def draw_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        '''
        print("Len of coord is {}".format(len(coords)))
        for coord in coords:
            cv2.rectangle(image,(coord[0],coord[1]),(coord[2],coord[3]),(0,255,0),1)
        return image

    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you
        '''

        global threshold, initial_h, initial_w
        arr = outputs.flatten()
        matrix = np.reshape(arr,(-1,7))
        matrix = [item for item in matrix if item[2]>threshold]
        if len(matrix)>0:
            *matrix, = map(lambda x: x[3:7],matrix)
            matrix=np.array(matrix)
            matrix[:,0] = matrix[:,0]*initial_w
            matrix[:,2] = matrix[:,2]*initial_w
            matrix[:,1] = matrix[:,1]*initial_h
            matrix[:,3] = matrix[:,3]*initial_h
            *matrix, = map(lambda x: list(map(int,x)),matrix)
        print("Matrix created")
        return matrix

    def preprocess_input(self, image):
        '''
        Adecute the image to be processed
        '''
        self.input_blob = next(iter(self.model.inputs))
        shape = self.model.inputs[self.input_blob].shape
        frame = cv2.resize(image,(shape[3],shape[2]))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape(1, *frame.shape)
        print("Image preprocessed successfully")        
        return frame

def main(args):
    global threshold, initial_w, initial_h
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()
    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text="Queue Monitoring   "
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')
        out_video.release()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)
