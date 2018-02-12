import cv2
import csv
import numpy as np
import os
import sklearn


	
def getImagePathsAndCorrectedMeasurements(dataPath, correction):
    """
    @author : Mohit Ak
    @description:     This module gets center, left, right images merges them into a single array. It also created a 
    measurement array for the left and right images by adding the correction
    @parameters:  
    data path -  Location of the data directory
    correction - The correction to be applied to the left and right images
    @return:   - Merged imagePaths and measurements
    """
    folders = [x[0] for x in os.walk(dataPath)]
    print("folders",folders)
    dataFolders = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), folders))
    print("dataFolders",dataFolders)
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    print(dataFolders)
    for folder in dataFolders:
        lines = []
        with open(folder + '/driving_log.csv') as csvFile:
            reader = csv.reader(csvFile)
            for line in reader:
                lines.append(line)

        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            # print(line[3])
            measurements.append(float(line[3]))
            # print("line",line)
            center.append(dataPath +'/IMG/'+line[0].split('/')[-1].strip())
            left.append(dataPath +'/IMG/'+line[1].split('/')[-1].strip())
            right.append(dataPath +'/IMG/'+line[2].split('/')[-1].strip())
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)

    imagePaths = []
    imagePaths.extend(centerTotal)
    imagePaths.extend(leftTotal)
    imagePaths.extend(rightTotal)
    measurements_corrected = []
    measurements_corrected.extend(measurementTotal)
    measurements_corrected.extend([x + correction for x in measurementTotal])
    measurements_corrected.extend([x - correction for x in measurementTotal])
    return (imagePaths, measurements)

def generator(samples, batch_size=32):
    """
    @author : Mohit Ak
    @description :      Generates required images and measurement in batches of batch size where the previous lists are retained for future calls
    @parameter : 
    samples -    list of objects containing imagePath and measuremnet
    @param batch_size: batch size to generate data
    """
    samples_count = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, samples_count, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                #Note - Very important step as the images are always input in RGB and cv returns VGR
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Data augmentation by Flipping images on Demand so that they are not saved unnecessarily
                images.append(cv2.flip(image,1))
                # Measurement flipped by multiplying with -1
                angles.append(measurement*-1.0)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)