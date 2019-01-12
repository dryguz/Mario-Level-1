import cv2


def run_transfer(x, net):

    # grab the frame from the threaded video stream
    frame = x

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    #frame = resize(frame, width=600)
    orig = frame.copy()
    (h, w) = frame.shape[:2]

    # construct a blob from the frame, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
                                 (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    # show the original frame along with the output neural style
    # transfer
    #cv2.imshow("Input", frame)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized