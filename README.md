
# Skin Disease Detection
Skin Disease Detection (namely **Skin-Genie**) is a machine learning project aimed at identifying various skin diseases using image classification techniques. This project uses convolutional neural networks (CNNs) to analyze and classify images of skin conditions, helping in early diagnosis and treatment.

List of skin diseases identified by this project includes: *Eczema*, *Melanoma*, *Atopic Dermatitis*, *Basal Cell Carcinoma (BCC)*, *Melanocytic Nevi (NV)*, *Benign Keratosis-like Lesions (BKL)*, *Psoriasis Lichen Planus*, *Seborrheic Keratoses*, *Tinea Ringworm Candidiasis*, and *Warts Molluscum*.

## Features

* User Authentication (Signin and Signup) 
* Detect skin diseases via *webcam* or *upload an image*. 
* After successful detection, *disease name*, *accuracy*, and *recommended medicine* are shown.
* Also, there is an option to find *nearby clinics / medical shops*, that will show your current location.

## Installation
**1\. Clone the repository:**
```bash
git clone https://github.com/rahul-singh03/Skin-Disease-Detection.git
cd Skin-Disease-Detection
```
**2\. Install the required packages:**
```bash
pip install -r requirements.txt
```
**3\. External requirements:**

3.1. **Install MongoDB:**   
Download the MongoDB installer from the [official MongoDB download center](https://www.mongodb.com/try/download/community) and follow the installation instructions.
 
3.2. **Start MongoDB Server:**
* Open the Command Prompt as an administrator.
* Navigate to the MongoDB installation directory:
```bash 
cd C:\Program Files\MongoDB\Server\<version>\bin
``` 
Replace <version> with your MongoDB version number.  
* Start the MongoDB server by running:
```bash
mongod
```
## Usage
1\. Run the `app.py` script:
```bash
python app.py
```
> **NOTE:**  
> Before running the `app.py`, You must have the `model.keras` file.  
> To generate the file, You have to run the `trainDataset.py` script. 
2\. The output will be generated and the development server will be started.  
3\. Press `Ctrl+C` in the terminal to quit the application.

## Dataset
The project utilizes a pre-labeled dataset of skin diseases from [Kaggle](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset "Skin Diseases Image Dataset") to train the model. Ensure the dataset is correctly formatted and placed in the appropriate directory before training.  
> **NOTE:**  
> To run the script `trainDataset.py`, place the downloaded dataset in the current folder, that is parted into `image_classes` and `test` folders. 

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

##  
For more detailed information, refer to the project report and presentation included in the repository.  
[Project Report (PDF)](Project%20Report.pdf)  
[Project Presentation (PPTX)](Project%20Presentation.pptx)


