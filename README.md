# Signature Verification Using Python
# Gradients
Comparison of two images using gradient vectors

- Using method 1 with 4 chain code
- Using method 2 with 32 chain code

Perform following on the image:
1. Split image matrix into 9x9 blocks

Perform following on each pixel in a block
1. Calculate gradient in x and y direction (gx and gy)
2. Calculate intensity and angle based on gx and gy
3. Calculate 4 chain code based on intensity and angle

Sum up all the 4 chain code of all pixels in a block to get one 4 chain code value for a block

Image 1 has 9x9 4 chain code values
Image 2 has 9x9 4 chain code values

Compare the two sets of 9x9 4 chain code values to determine similarity
