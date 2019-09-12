""" This code uses 32 chain code to compare two images """

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# Load the image in grayscale
img1 = cv.imread('obama1.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('obama2.png', cv.IMREAD_GRAYSCALE)

# Calculate the number of rows and cols in the image matrix
num_of_blocks = 9

# Store all the individual blocks in this array
def split_image(img, num_of_blocks):
    rows, cols = img.shape
    block_rows = rows//num_of_blocks
    block_cols = cols//num_of_blocks
    blocks = []
    # Divide the image into blocks
    for row in range(0, rows - block_rows + 1, block_rows):
        for col in range(0, cols -block_cols + 1, block_cols):
            # Print statement so that people don't get bored waiting
            print("block is being made...")
            block = np.zeros((block_rows, block_cols))
            for i in range(0, block_rows):
                for j in range(0, block_cols):
                    block[i][j] = img[row + i][col + j]
            blocks.append(block)
    return blocks

blocks1 = split_image(img1, num_of_blocks)
blocks2 = split_image(img2, num_of_blocks)

'''
fig1 = plt.figure()
fig1.suptitle('Oprah1')
for i in range(0, len(blocks1)):
    ax = fig1.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(blocks1[i], cmap = 'gray')

fig2 = plt.figure()
fig2.suptitle('Oprah2')
for i in range(0, len(blocks2)):
    ax = fig2.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(blocks2[i], cmap = 'gray')

plt.show()
'''

def calculate_gradients(blocks):
    # Calculate gx and gy
    gx = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)])
    gy = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)])

    gx_blocks = []
    gy_blocks = []

    for i in range(0, len(blocks)):
        block = blocks[i]

        block_rows, block_cols = blocks[i].shape

        gx_block = np.zeros((block_rows, block_cols))
        gy_block = np.zeros((block_rows, block_cols))

        print("Calculating gx for block {}".format(i + 1))
        for row in range(1, block_rows - 1):
            for col in range(1, block_cols - 1):
                for filter_row in range(0, 3):
                    for filter_col in range(0, 3):
                        gx_block[row][col] += (block[row + (filter_row - 1)][col + (filter_col - 1)] * gx[filter_row][filter_col])

        print("Calculating gy for block {}".format(i + 1))
        for row in range(1, block_rows - 1):
            for col in range(1, block_cols - 1):
                for filter_row in range(0, 3):
                    for filter_col in range(0, 3):
                        gy_block[row][col] += (block[row + (filter_row - 1)][col + (filter_col - 1)] * gy[filter_row][filter_col])

        gx_blocks.append(gx_block)
        gy_blocks.append(gy_block)

    return gx_blocks, gy_blocks

gx_blocks1, gy_blocks1 = calculate_gradients(blocks1)
gx_blocks2, gy_blocks2 = calculate_gradients(blocks2)

'''
fig1 = plt.figure()
fig1.suptitle('gx values for img1')
for i in range(0, len(gx_blocks1)):
    ax = fig1.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(gx_blocks1[i], cmap = 'gray')

fig2 = plt.figure()
fig2.suptitle('gy values for img1')
for i in range(0, len(gy_blocks1)):
    ax = fig2.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(gy_blocks1[i], cmap = 'gray')

fig3 = plt.figure()
fig3.suptitle('gx values for img2')
for i in range(0, len(gx_blocks2)):
    ax = fig3.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(gx_blocks2[i], cmap = 'gray')

fig4 = plt.figure()
fig4.suptitle('gy values for img2')
for i in range(0, len(gy_blocks2)):
    ax = fig4.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(gy_blocks2[i], cmap = 'gray')

plt.show()
'''

def calculate_intensities(gx_blocks, gy_blocks):
    Q_blocks = []
    for i in range(0, len(gx_blocks)):
        gx_block = gx_blocks[i]
        gy_block = gy_blocks[i]
        block_rows, block_cols = gx_block.shape
        print("Calculating Q for block {}".format(i + 1))
        Q_block = np.zeros((block_rows, block_cols))
        for row in range(1, block_rows - 1):
            for col in range(1, block_cols - 1):
                Q_block[row][col] = np.sqrt((gx_block[row][col]**2) + (gy_block[row][col]**2))
        Q_blocks.append(Q_block)
    return Q_blocks

Q_blocks1 = calculate_intensities(gx_blocks1, gy_blocks1)
Q_blocks2 = calculate_intensities(gx_blocks2, gy_blocks2)

'''
fig1 = plt.figure()
fig1.suptitle('Intensities for img1')
for i in range(0, len(Q_blocks1)):
    ax = fig1.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(Q_blocks1[i], cmap = 'gray')

fig2 = plt.figure()
fig2.suptitle('Intensities for img2')
for i in range(0, len(Q_blocks2)):
    ax = fig2.add_subplot(num_of_blocks, num_of_blocks, i + 1)
    ax.imshow(Q_blocks2[i], cmap = 'gray')

plt.show()
'''

def calculate_chain32_code(gx_blocks, gy_blocks, Q_blocks):
    chain32_code_blocks = []

    split_factor = 2*np.pi/32

    for i in range(0, len(Q_blocks)):
        gx_block = gx_blocks[i]
        gy_block = gy_blocks[i]
        Q_block = Q_blocks[i]

        block_rows, block_cols = Q_block.shape

        print("Calculating theta for block {}".format(i + 1))
        
        chain32_code_block = np.zeros((block_rows, block_cols), dtype = (float, 32))

        for row in range(1, block_rows - 1):
            for col in range(1, block_cols - 1):
                gx_val = gx_block[row][col]
                gy_val = gy_block[row][col]
                Q_val = Q_block[row][col]

                if (gx_val > 0) and (gy_val > 0):
                    #The point lies in the first quadrant
                    theta = np.arctan2(gy_val, gx_val)
                    index_val = int(theta/(2*np.pi/32))
                    # Create a tuple of 32 values of all zero
                    my_tuple = (0, ) * 32
                    # Tuples are immutable, therefore create a list then modify it
                    my_tuple = list(my_tuple)
                    my_tuple[index_val] = Q_val
                    my_tuple = tuple(my_tuple)
                    # Assign the non-component value
                    chain32_code_block[row][col] = my_tuple

                elif (gx_val < 0) and (gy_val > 0):
                    # The point lies in the second quadrant
                    theta = np.arctan2(gy_val, -gx_val) + (np.pi/2)
                    index_val = int(theta/(2*np.pi/32))
                    # Create a tuple of 32 values of all zero
                    my_tuple = (0, ) * 32
                    # Tuples are immutable, therefore create a list then modify it
                    my_tuple = list(my_tuple)
                    my_tuple[index_val] = Q_val
                    my_tuple = tuple(my_tuple)
                    # Assign the non-component value
                    chain32_code_block[row][col] = my_tuple

                elif (gx_val < 0) and (gy_val < 0):
                    # The point lies in the third quadrant
                    theta = np.arctan2(-gy_val, -gx_val) + np.pi
                    index_val = int(theta/(2*np.pi/32))
                    # Create a tuple of 32 values of all zero
                    my_tuple = (0, ) * 32
                    # Tuples are immutable, therefore create a list then modify it
                    my_tuple = list(my_tuple)
                    my_tuple[index_val] = Q_val
                    my_tuple = tuple(my_tuple)
                    # Assign the non-component value
                    chain32_code_block[row][col] = my_tuple

                elif (gx_val > 0) and (gy_val < 0):
                    # The point lies in the fourth quadrant
                    theta = np.arctan2(-gy_val, gx_val) + 3*(np.pi/2)
                    index_val = int(theta/(2*np.pi/32))
                    # Create a tuple of 32 values of all zero
                    my_tuple = (0, ) * 32
                    # Tuples are immutable, therefore create a list then modify it
                    my_tuple = list(my_tuple)
                    my_tuple[index_val] = Q_val
                    my_tuple = tuple(my_tuple)
                    # Assign the non-component value
                    chain32_code_block[row][col] = my_tuple
        
        chain32_code_blocks.append(chain32_code_block)
    return chain32_code_blocks

chain32_code_blocks1 = calculate_chain32_code(gx_blocks1, gy_blocks1, Q_blocks1)
chain32_code_blocks2 = calculate_chain32_code(gx_blocks2, gy_blocks2, Q_blocks2)

def calculate_chain32_code_sum(chain32_code_blocks):
    chain32_code_sum_blocks = []
    for i in range(0, len(chain32_code_blocks)):
        #Add the 4 chain code of every pixel in each block
        chain32_code_block = chain32_code_blocks[i]
        block_rows, block_cols, tuple_length = chain32_code_block.shape
        total_sum = (0.0,) * 32
        for row in range(0, block_rows):
            for col in range(0, block_cols):
                total_sum = tuple(map(lambda x,y : x + y, total_sum, chain32_code_block[row][col]))
        chain32_code_sum_blocks.append(total_sum)
    return chain32_code_sum_blocks
    
chain32_code_sum_blocks1 = calculate_chain32_code_sum(chain32_code_blocks1)
chain32_code_sum_blocks2 = calculate_chain32_code_sum(chain32_code_blocks2)

'''
print(chain32_code_sum_blocks1)
print(chain32_code_sum_blocks2)
'''

def calculate_chain32_code_grand_sum(chain32_code_sum_blocks):
    chain32_code_grand_sum = chain32_code_sum_blocks[0]
    # Sum up all the 32 chain codes of each block
    for i in range(1, len(chain32_code_sum_blocks)):
        #chain32_code_grand_sum = tuple(map(lambda x,y : x+y, chain32_code_grand_sum, chain32_code_sum_blocks[i]))
        chain32_code_grand_sum += chain32_code_sum_blocks[i]
    return chain32_code_grand_sum

chain32_code_grand_sum1 = calculate_chain32_code_grand_sum(chain32_code_sum_blocks1)
chain32_code_grand_sum2 = calculate_chain32_code_grand_sum(chain32_code_sum_blocks2)

'''
print(chain32_code_grand_sum1)
print(chain32_code_grand_sum2)
'''

def calculate_score(chain32_code_grand_sum1, chain32_code_grand_sum2):
    numerator = 0
    for i in range(0, len(chain32_code_grand_sum1)):
        numerator += chain32_code_grand_sum1[i]*chain32_code_grand_sum2[i]
    sum_of_squares1 = 0
    sum_of_squares2 = 0
    for i in range(0, len(chain32_code_grand_sum1)):
        sum_of_squares1 += chain32_code_grand_sum1[i]**2
        sum_of_squares2 += chain32_code_grand_sum2[i]**2
    denominator = np.sqrt(sum_of_squares1) * np.sqrt(sum_of_squares2)
    return numerator/denominator

print(calculate_score(chain32_code_grand_sum1, chain32_code_grand_sum2))
