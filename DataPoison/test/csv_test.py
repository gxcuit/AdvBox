import unittest
from PIL import Image, ImageOps
from utils import CSVUtils


class MyTestCase(unittest.TestCase):
    def test_something(self):
        image = Image.open('../mnist/MNIST/raw/poison/x.jpg')
        image=image.convert('L')
        invert = ImageOps.invert(image)
        invert.save('../mnist/MNIST/raw/poison/x_inv.jpg')

    def test_eval(self):
        CSVUtils.eval_accu('./data.csv')
        CSVUtils.get_csv_image('../mnist/MNIST/raw/csv/mnist_one_pixel_test.csv',16)

if __name__ == '__main__':
    unittest.main()
