# 5C-Networks
Brain MRI Metastasis Segmentation This project demonstrates proficiency in computer vision by implementing and comparing two architectures—Nested U-Net (U-Net++) and Attention U-Net—for brain MRI metastasis segmentation. A web application is also developed to showcase the best-performing model.

Dataset The dataset provided includes Brain MRI images and their corresponding metastasis segmentation masks. Images without corresponding masks and vice versa are ignored. The dataset is split into an 80% training set and a 20% testing set.

Download the dataset from the following link: Data.zip.

Data Preprocessing CLAHE Preprocessing: Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to enhance metastasis visibility in the MRI images. Normalization and Augmentation: The dataset undergoes normalization and augmentation, including rotations, flips, and zooms to account for the challenges of metastasis segmentation. Model Implementation We implemented two different architectures for metastasis segmentation:

Nested U-Net (U-Net++): A variation of U-Net with dense skip connections and nested sub-networks that provide deeper feature representations, improving segmentation accuracy. Attention U-Net: Introduces attention gates to allow the model to focus on metastasis regions, enhancing the segmentation accuracy by highlighting relevant features. Model Training and Evaluation Both models were trained on the preprocessed dataset, with the DICE Score as the primary evaluation metric for segmentation accuracy.

Web Application Development We developed a web application with the following components:

Backend: Built with FAST API to serve the best-performing metastasis segmentation model. Frontend: Developed with Streamlit to allow users to upload brain MRI images and view the metastasis segmentation results.

Conclusion This project demonstrates the effectiveness of both Nested U-Net and Attention U-Net architectures in brain MRI metastasis segmentation. The models and web application provide a useful framework for future research and potential clinical applications.
