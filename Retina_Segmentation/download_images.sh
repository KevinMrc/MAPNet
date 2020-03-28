# Download images
wget  --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17wVfELqgwbp4Q02GD247jJyjq6lwB0l6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17wVfELqgwbp4Q02GD247jJyjq6lwB0l6" -O DRIVE.zip && rm -rf /tmp/cookies.txt
unzip DRIVE.zip
rm DRIVE.zip

# Preprocess images
python prepare_datasets_DRIVE.py

# Creates folders
mkdir image
mkdir mask
mkdir image/train
mkdir mask/train
mkdir image/test
mkdir mask/test

# Creates patches
python save_patch.py --patch_size 256 --n_patches 1000

# Delete temp files
rm extract_patches.py
rm help_functions.py
rm pre_processing.py
rm prepare_datasets_DRIVE.py
rm save_patch.py
rm -R DRIVE
#rm -R DRIVE_datasets_training_testing

# Zip folders
#zip -r image_64_sep.zip image 
#zip -r mask_64_sep.zip mask
