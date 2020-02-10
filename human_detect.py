'''
human_detect.py
- Zhen Tang

Uses HOG (Hisograms of Oriented Gradients) and LBP (Local Binary Pattern)
features to detect humans in images
'''

from PIL import Image
import numpy as np
import math
import os
import argparse

# gradient operation constants
SOBEL_X_MASK = np.array([[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]])
SOBEL_Y_MASK = np.array([[ 1,  2,  1],
                         [ 0,  0,  0],
                         [-1, -2, -1]])
SOBEL_MASK_NORM = 4

# HOG related constants
HOG_CELL_SIZE = 8
BIN_DIFFERENCE = 20
HOG_FEATURE_SIZE = 7524
HOG_BLOCK_HEIGHT = 19
HOG_BLOCK_WIDTH = 11
HOG_BLOCK_SIZE = 2
HOG_BIN_TOTAL = 9

# for calculating LBP
LBP_POWER_ARRAY = np.array([[128,  64,  32],
                            [  1,   0,  16],
                            [  2,   4,   8]])
LBP_FEATURE_SIZE = 3540
LBP_BLOCK_SIZE = 16
LBP_BLOCK_HEIGHT = 10
LBP_BLOCK_WIDTH = 6
LBP_BIN_TOTAL = 59

LEARNING_RATE = 0.1

# convertToGrayScale(img_arr)
# - converts an image to grayscale based on its RGB values
def convertToGrayscale(img_arr):
    result = np.zeros((img_arr.shape[0], img_arr.shape[1]), np.uint8)
    
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            # I = Round(0.299R + 0.587G + 0.114B)
            result[i, j] = np.rint(0.299 * img_arr[i, j, 0] 
                                   + 0.587 * img_arr[i, j, 1] 
                                   + 0.114 * img_arr[i, j, 2]).astype(np.uint8)
    
    return result

# applyMask() - applies the mask to the image centered at the location
def applyMask(img_arr, pixel_i, pixel_j, mask):
    start_i = pixel_i - int(mask.shape[0] / 2)
    start_j = pixel_j - int(mask.shape[1] / 2)
    
    # return 0 if undefined due to out of bound
    if (start_i < 0 or start_j < 0
       or (pixel_i + int(mask.shape[0] / 2)) >= img_arr.shape[0]
       or (pixel_j + int(mask.shape[1] / 2)) >= img_arr.shape[1]):
        return 0
    
    result = 0
    
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            # return 0 if mask lies in undefined region
            if (img_arr[start_i + i, start_j + j] == 0):
                return 0
            
            result += (mask[i, j] * img_arr[start_i + i, start_j + j])
    
    return result

# convolution() - performs discrete convolution with the mask
def convolution(img_arr, mask):
    result = np.zeros_like(img_arr).astype(float)

    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            result[i, j] = applyMask(img_arr, i, j, mask)

    return result

# gradientResponse() - calculate vertical and horizontal gradient responses
def gradientResponse(img_arr):
    horizontal = convolution(img_arr, SOBEL_X_MASK) / SOBEL_MASK_NORM
    vertical = convolution(img_arr, SOBEL_Y_MASK) / SOBEL_MASK_NORM
    return (horizontal, vertical)

def gradientMagnitude(horizontal, vertical):
    magnitude = np.empty_like(horizontal)
    
    for i in range(0, horizontal.shape[0]):
        for j in range(0, horizontal.shape[1]):
            # output is 0 if mask lies in undefined region
            if (horizontal[i, j] == 0 or vertical[i, j] == 0):
                magnitude[i, j] = 0
            else:
                horizontal_square = horizontal[i, j] * horizontal[i, j]
                vertical_square = vertical[i, j] * vertical[i, j]
                magnitude[i, j] = math.sqrt(horizontal_square + vertical_square)
    
    # normalizes the magnitude
    # maximum possible value of magnitude is sqrt(2*255*255)
    # normalization factor of sqrt(2)
    magnitude /= math.sqrt(2)
    magnitude = np.rint(magnitude).astype(np.uint8)
    
    return magnitude

# gradientAngle() - computed as arctan(gradient_y / gradient_x)
def gradientAngle(horizontal, vertical):
    angle = np.empty_like(horizontal)

    for i in range(0, horizontal.shape[0]):
        for j in range(0, horizontal.shape[1]):
            # no angle if both horizontal or vertical gradient are 0
            if (horizontal[i, j] == 0 or vertical[i, j] == 0):
                angle[i, j] = 0
            # math.atan2 handles cases in which the horizontal gradient is 0
            else:
                angle[i, j] = math.atan2(vertical[i, j], horizontal[i, j]) * 180 / math.pi

    return angle

# cellHistogramHOG() - compute the histogram for a cell
def cellHistogramHOG(feature_vector, feature_pos, init_i, init_j, magnitude_arr, angle_arr):
    for i in range(init_i, init_i + HOG_CELL_SIZE):
        for j in range(init_j, init_j + HOG_CELL_SIZE):
            current_angle = angle_arr[i, j] % 180
            
            # bin number offset by 1 for indices
            smaller_bin = int(current_angle / 20)
            bigger_bin = 0 if smaller_bin == 8 else smaller_bin + 1
            
            smaller_center = smaller_bin * 20
            bigger_center = bigger_bin * 20
            
            # split magnitude based on distance to center
            smaller_bin_split = 1 - ((current_angle - smaller_center) / BIN_DIFFERENCE)
            bigger_bin_split = 1 - (((bigger_center - current_angle) % 180) / BIN_DIFFERENCE)
                
            feature_vector[feature_pos + smaller_bin] += magnitude_arr[i, j] * smaller_bin_split
            feature_vector[feature_pos + bigger_bin] += magnitude_arr[i, j] * bigger_bin_split
            

# computeHOGFeatures - loops the blocks, cells and computeHistogramHOG() for each cell
def computeHOGFeature(magnitude, angle):
    feature_vector = np.zeros(HOG_FEATURE_SIZE)
    feature_position = 0
    
    # loop blocks
    for block_i in range(0, HOG_BLOCK_HEIGHT):
        for block_j in range(0, HOG_BLOCK_WIDTH):
            pixel_i = block_i * HOG_CELL_SIZE
            pixel_j = block_j * HOG_CELL_SIZE
            
            # loop cells
            for cell_i in range(0, HOG_BLOCK_SIZE):
                for cell_j in range(0, HOG_BLOCK_SIZE):
                    init_cell_pixel_i = pixel_i + cell_i * HOG_CELL_SIZE
                    init_cell_pixel_j = pixel_j + cell_j * HOG_CELL_SIZE
                    
                    cellHistogramHOG(feature_vector, feature_position, 
                                     init_cell_pixel_i, init_cell_pixel_j, 
                                     magnitude, angle)
                    
                    feature_position += HOG_BIN_TOTAL
                    
    return feature_vector

# L2NormVector 
# - Normalize the vector by its blockwise L2 value
# - each block occupies a length of "dimension"
def L2NormVector(vector):
    dimension = HOG_BLOCK_SIZE * HOG_BLOCK_SIZE * HOG_BIN_TOTAL
    block_num = HOG_BLOCK_HEIGHT * HOG_BLOCK_WIDTH
    
    for i in range(0, block_num):
        l2_norm = np.sqrt(np.sum(np.square(vector[i * dimension : (i + 1) * dimension])))
        if (l2_norm == 0):
            vector[i * dimension : (i + 1) * dimension] = 0
        else:
            vector[i * dimension : (i + 1) * dimension] /= l2_norm

# calculateLBP() - return the LBP value at i, j
def calculateLBP(magnitude, center_i, center_j):
    center_value = magnitude[center_i, center_j]
    top_left_i = center_i - 1
    top_left_j = center_j - 1
    
    result = 0
    for i in range(0, 3):
        for j in range(0, 3):
            # out of bound check
            if (top_left_i + i < 0 or top_left_i + i >= magnitude.shape[0]) \
                or (top_left_j + j < 0 or top_left_j + j >= magnitude.shape[1]):
                return 5
            else:
                compareTo = magnitude[top_left_i + i, top_left_j + j]
            if (center_value < compareTo):
                result += LBP_POWER_ARRAY[i, j]
    return result

# blockHistogramLBP() 
# - loops the pixels in the block
# - increment the appropriate bins based on block number and pixel LBP value
def blockHistogramLBP(feature_vector, feature_pos, 
                      init_i, init_j, magnitude):
    for i in range(init_i, init_i + 16):
        for j in range(init_j, init_j + 16):
            value = calculateLBP(magnitude, i, j)
            
            # not really sure how to simplify this part
            if (value == 0): bin_num = 0
            elif (value == 1): bin_num = 1
            elif (value == 2): bin_num = 2
            elif (value == 3): bin_num = 3
            elif (value == 4): bin_num = 4
            elif (value == 6): bin_num = 5
            elif (value == 7): bin_num = 6
            elif (value == 8): bin_num = 7
            elif (value == 12): bin_num = 8
            elif (value == 14): bin_num = 9
            elif (value == 15): bin_num = 10
            elif (value == 16): bin_num = 11
            elif (value == 24): bin_num = 12
            elif (value == 28): bin_num = 13
            elif (value == 30): bin_num = 14
            elif (value == 31): bin_num = 15
            elif (value == 32): bin_num = 16
            elif (value == 48): bin_num = 17
            elif (value == 56): bin_num = 18
            elif (value == 60): bin_num = 19
            elif (value == 62): bin_num = 20
            elif (value == 63): bin_num = 21
            elif (value == 64): bin_num = 22
            elif (value == 96): bin_num = 23
            elif (value == 112): bin_num = 24
            elif (value == 120): bin_num = 25
            elif (value == 124): bin_num = 26
            elif (value == 126): bin_num = 27
            elif (value == 127): bin_num = 28
            elif (value == 128): bin_num = 29
            elif (value == 129): bin_num = 30
            elif (value == 131): bin_num = 31
            elif (value == 135): bin_num = 32
            elif (value == 143): bin_num = 33
            elif (value == 159): bin_num = 34
            elif (value == 191): bin_num = 35
            elif (value == 192): bin_num = 36
            elif (value == 193): bin_num = 37
            elif (value == 195): bin_num = 38
            elif (value == 199): bin_num = 39
            elif (value == 207): bin_num = 40
            elif (value == 223): bin_num = 41
            elif (value == 224): bin_num = 42
            elif (value == 225): bin_num = 43
            elif (value == 227): bin_num = 44
            elif (value == 231): bin_num = 45
            elif (value == 239): bin_num = 46
            elif (value == 240): bin_num = 47
            elif (value == 241): bin_num = 48
            elif (value == 243): bin_num = 49
            elif (value == 247): bin_num = 50
            elif (value == 248): bin_num = 51
            elif (value == 249): bin_num = 52
            elif (value == 251): bin_num = 53
            elif (value == 252): bin_num = 54
            elif (value == 253): bin_num = 55
            elif (value == 254): bin_num = 56
            elif (value == 255): bin_num = 57
            else: bin_num = 58
                
            feature_vector[feature_pos + bin_num] += 1

# computeLBPFeature()
# - compute the LBP feature for each block
#   from left to right then top to bottom
# - normalizes the histogram within each block
#   by the total number of pixels in each block
def computeLBPFeatures(magnitude):
    feature_vector = np.zeros(LBP_FEATURE_SIZE)
    feature_position = 0
    
    # loop the blocks
    for i in range(0, LBP_BLOCK_HEIGHT):
        for j in range(0, LBP_BLOCK_WIDTH):
            init_i = i * LBP_BLOCK_SIZE
            init_j = j * LBP_BLOCK_SIZE
            
            blockHistogramLBP(feature_vector, feature_position,
                              init_i, init_j, magnitude)
            
            feature_position += 59
            
    # normalize LBP by number of pixels in each block (16*16 = 256)
    return feature_vector / 256

# forwardPropagate()
# - forward propagation through the neural network
def forwardPropagate(features, input_weights, hidden_weights):
    featuresWithBias = np.append([[-1]], features, 0)
    hiddenIn = input_weights.dot(featuresWithBias)
    # ReLU activation function
    hiddenOut = np.maximum(0, hiddenIn)
    
    hiddenOutWithBias = np.append([[-1]], hiddenOut, 0)
    predict = hidden_weights.dot(hiddenOutWithBias)[0, 0]
    # sigmoid activation function
    # setting an upper and lower bound because the result was being rounded
    # which causes the change to weight in back propagation to be 0
    # as derivative of sigmoid is prediction*(1-prediction)
    # which is 0 in the case that the prediction gets rounded to 1 or 0
    predict = np.minimum(1 / (1 + np.exp(-predict)), 0.99999999999999)
    predict = np.maximum(predict, 0.00000000000001)
    return (predict, hiddenOut)

# backPropagate()
# - back propagate through the neural network and
#   update the weight values
def backPropagate(learning_rate, prediction, expected, 
                  features, weight_input, weight_hidden, 
                  hidden_vector):
    # output to hidden layer
    err_i = expected - prediction
    sigmoid_derivative = prediction * (1 - prediction)
    delta_i = err_i * sigmoid_derivative
    hiddenWithBias = np.append([[-1]], hidden_vector, 0)
    new_hidden_w = weight_hidden + learning_rate * hiddenWithBias.transpose() * delta_i
    
    # hidden layer to input
    relu_derivative = (hidden_vector > 0).astype(int)
    err_j = new_hidden_w.transpose()[1:] * delta_i
    delta_j = relu_derivative * err_j
    
    featuresWithBias = np.append([[-1]], features, 0)
    new_input_w = weight_input + learning_rate * np.dot(delta_j, featuresWithBias.transpose())
    
    return (new_input_w, new_hidden_w)

# computeFeatureVector()
# - compute the following for the file "filename"
#       HOG-only feature vector if hog_lbp is False
#       HOG-LBP feature vector if hog_lbp is True
def computeFeatureVector(filename, hog_lbp):
    im = Image.open(filename) 
    image_arr = np.asarray(im)
    im.close()
    gray = convertToGrayscale(image_arr)
        
    # gradient vectors
    (gradient_x, gradient_y) = gradientResponse(gray)
    img_magnitude = gradientMagnitude(gradient_x, gradient_y)
    img_angle = gradientAngle(gradient_x, gradient_y)
        
    # HOG
    features = computeHOGFeature(img_magnitude, img_angle)
    L2NormVector(features)
        
    # LBP
    if (hog_lbp):
        lbp_features = computeLBPFeatures(img_magnitude)
        features = np.hstack((features, lbp_features))
    
    return features

# trainNeuralNet()
# - train the neural network using the supplied information
def trainNeuralNet(training_files, hidden_size, learning_rate, hog_lbp):
    print("GETTING IMAGE FEATURES...")
    feature_vector = []
    # create array containing feature values
    for img_path in training_files:
        feature_vector.append(np.append(computeFeatureVector(img_path, hog_lbp), [training_files[img_path]]))
    feature_vector = np.array(feature_vector)
    
    # initialize random weights in the range (-0.05, 0.05)
    input_weights = np.random.uniform(-0.05, 0.05, (hidden_size, feature_vector[0].shape[0]))
    hidden_weights = np.random.uniform(-0.05, 0.05, (1, hidden_size + 1))
    
    # loop set up
    avg_error = 1
    epoch = 0
    
    print("STARTING TRAINING")
    
    while (avg_error > 0.005 and epoch < 1000):
        total_error = 0
        np.random.shuffle(feature_vector)
        
        # compute the error for each image and update the weight
        for img_feature in range(0, feature_vector.shape[0]):
            length = feature_vector[img_feature].shape[0]
            expected = feature_vector[img_feature][-1]
            training_feature = np.reshape(feature_vector[img_feature][:-1], (length - 1, 1))
            
            (predict, hidden_nodes) = forwardPropagate(training_feature, input_weights, hidden_weights)
            (input_weights, hidden_weights) = backPropagate(learning_rate, predict, expected, 
                                                            training_feature, input_weights, hidden_weights, 
                                                            hidden_nodes)
            total_error = total_error + abs(expected - predict)
        
        avg_error = total_error / feature_vector.shape[0]
        epoch = epoch + 1
        if (epoch % 10 == 0):
            print("EPOCH:", epoch, "\tAVG ERROR:", avg_error)
    
    print("FINAL EPOCH:", epoch, "\tAVG ERROR:", avg_error)
    return (input_weights, hidden_weights)

# testNeuralNet()
# - test the neural network on the test_files
def testNeuralNet(test_files, input_weights, hidden_weights, hog_lbp):
    feature_vector = []
    total_error = 0
    # create array containing feature values
    for img_path in test_files:
        features = computeFeatureVector(img_path, hog_lbp)
        features = np.reshape(features, (features.shape[0], 1))
            
        (predict, hidden_nodes) = forwardPropagate(features, input_weights, hidden_weights)
        
        if predict >= 0.6: classification = "human"
        elif predict <= 0.4: classification = "no-human"
        else: classification = "borderline"
            
        total_error = total_error + abs(test_files[img_path] - predict)
            
        print("File:", img_path, "\n\tPerceptron Output:", predict, "\n\tClassification", classification)
        
    print("AVERAGE ERROR:", total_error / len(test_files))

def main():
    # using a seed for consistent results
    np.random.seed(6643)

    # argument parser
    parser = argparse.ArgumentParser(description='Uses HOG and LBP features to detect human in images')
    parser.add_argument('hidden_size', help='size of hidden layer', type=int)
    parser.add_argument('-l', '--lbp', action='store_true', help='Use both HOG and LBP if this is set')

    args = parser.parse_args()

    # file preparations
    training_neg_dir = "Image Data/Training images (Neg)"
    training_pos_dir = "Image Data/Training images (Pos)"
    test_neg_dir = "Image Data/Test images (Neg)"
    test_pos_dir = "Image Data/Test images (Pos)"

    training_files = {}
    for root, dirs, files in os.walk(training_neg_dir):
        for file in files:
            training_files[training_neg_dir + "/" + file] = 0
    for root, dirs, files in os.walk(training_pos_dir):
        for file in files:
            training_files[training_pos_dir + "/" + file] = 1

    test_files = {}
    for root, dirs, files in os.walk(test_neg_dir):
        for file in files:
            test_files[test_neg_dir + "/" + file] = 0
    for root, dirs, files in os.walk(test_pos_dir):
        for file in files:
            test_files[test_pos_dir + "/" + file] = 1

    (input_w, hidden_w) = trainNeuralNet(training_files, args.hidden_size, LEARNING_RATE, args.lbp)

    testNeuralNet(test_files, input_w, hidden_w, args.lbp)

if __name__ == "__main__":
    main()