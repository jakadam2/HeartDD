import pydicom as dicom
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import matplotlib.pyplot as plt
from pydicom.data import get_testdata_file

tk.Tk().withdraw()


def main():

    #filename = askopenfilename()
    filename = get_testdata_file("CT_small.dcm")
    extension = filename.rsplit(".", 1)[1] 
    file = None
    match extension:
        case "dcm":
            file = dicom.dcmread(filename)
        case "png" | "jpg":
            file = Image.open(filename)
        case _:
            print("Incorrect file format, files must be dicom, png or jpg")
            return
    if file == None:
        print("OwO, something went howwibly bad. Sowwwwy T~T")
        return  
    plt.imshow(file.pixel_array, cmap = plt.cm.gray)
    plt.show()



if __name__ == "__main__":
    main()