import os 
import subprocess 
import datetime

def convert_to_nii(list_of_dirs, parent_dir, out_dir, correct_tilt=True):
    # list_of_dirs should be a file with the names of directories containing
    # DICOM files that you want to convert, all directories should be in 
    # the same parent directory
    # example: 
    # case1
    # case2
    # ...
    # out_dir is the output PARENT directory, final output --> each case will 
    # be its own directory with the same name containing the .nii files 
    # make sure dcm2niix is in path 
    # correct_tilt specifies whether to use the tilted image of the non-til
    
    start_time = datetime.datetime.now() 
    with open(list_of_dirs, "r") as f:
        files = f.readlines() 

    files = [i.strip() for i in files]
    if os.path.exists(out_dir) is False:
        os.system("mkdir " + out_dir)
    for i in files: 
        temp_out = out_dir + "/" +  i
        if os.path.exists(temp_out) is False:
            os.system("mkdir " + temp_out)
        command = "dcm2niix -o " + temp_out + "- m y " + parent_dir + "/" + i 
        os.system(command+"/")
        ls_output = subprocess.check_output("ls " + temp_out, shell=True) 
        ls_output = ls_output.split() 
        exceptions = [] 
        if len(ls_output) > 2: 
            exceptions.append(i) 
            continue
        min_length = min(ls_output, key=len)
        max_length = max(ls_output, key=len)
        if correct_tilt: 
            os.system("rm " + temp_out + "/" + min_length)
        else: 
            os.system("rm " + temp_out + "/" + max_length)
    
    for e in exceptions:
        print(e + " has more than 2 files. Please check and delete extraneous files manually.\n")
    print("DONE!")
    print("Converted " + str(len(files)) + " files in: " + str(datetime.datetime.now() - start_time))
     
def extract_3dsift_feat(nii_path, out_dir, mac=True):
    # nii_path should just be the out_dir specified in convert_to_nii
    # out_dir should be the desired output directory for all the .key files
    # make sure featExtract is in path 
    # mac argument is for whether you're using macOS
    
    start_time = datetime.datetime.now()
    files = subprocess.check_output("ls " + nii_path, shell=True)
    files = files.split() 
    if mac: ext = ".mac"
    else: ext = ".ubu"
    for i in files: 
        command = "featExtract" + ext + " -qto_xyz " + nii_path + "/" + i + "/*.nii"
        command += " " + out_dir + "/" + i + ".key"
    print("DONE!")
    print("Extracted 3D-SIFT features from " + str(len(files)) + " images in: " + str(datetime.datetime.now() - start_time))
