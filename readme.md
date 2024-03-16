# Vanishing Greens
THE PROBLEM

4.2% of the world's tree cover loss was between 1990 and 2020. By 2030, there may be only 10% of the world's rainforests left. India, 668,400 hectares of deforestation. 
And every minute 2400 trees are cut down so by the end of 2 minutes nearly 5000 tress will be cut down.
So we can imagine the level of problem we are dealing.

IDEA DESCRIPTION


The project aims to develop a ML-based system for automated deforestation detection and classification using satellite imagery. The system will utilize Convolutional Neural Networks (CNNs) for image classification.

The project also aims to highlight areas of significant change with red contours with the help of image processing techniques.
We have came up with two solutions to tackle the problem of deforestation :-

Deforestation type detection :
The project aims to develop a ML-based system for automated deforestation detection and classification using satellite imagery. The system will utilize Convolutional Neural Networks (CNNs) for image classification.

Deforestation analysis: 
The project also aims to highlight areas of significant change with red contours with the help of image processing techniques.


SOLUTION OVERVIEW

The project proposes a comprehensive solution for deforestation detection and red contour marking using a combination of ML and image processing techniques. The workflow can be summarized as follows:

Data Collection:   Gather satellite imagery datasets, including both current and historical images of forested areas.

Deforestation Classification (ML-based):   Utilize a Convolutional Neural Network (CNN) for image classification to distinguish between deforested and forested areas.Train the model on dataset, optimizing for accuracy and generalization.

Red Contour Marking (Image Processing):   Develop image processing algorithms to mark red contours around detected deforested areas.Use techniques such as color thresholding, edge detection analysis to highlight changes.

Integration:   Seamlessly integrate the deforestation classification and red contour marking components for a cohesive system.



TECH STACK


A.   Image Classification (Deep Learning Model):
     Tensor Flow
     TensorFlow Image Preprocessing
     Convolutional Neural Network (CNN)
     TensorFlow Keras
     NumPy
     Matplotlib

B.   Deforestation Analysis:
     Open CV
     NumPy
     Matplotlib


The entire code is written in Python, leveraging its rich ecosystem for deep learning, image processing, and data analysis.

With the help of the following languages and libraries, all work carried out will be integrated into a website.


     HTML
     CSS
     JavaScript
     Svelte
     Tailwind
     Flask


IMPLEMENTATION


So basically we have planned to build a machine learning model which is trained on satellite imagery dataset to detect the type of deforestation.

Next the model will be trained and saved using the .keras format which store the architecture of the layers.

Once trained the model will be tested and evaluated on different images. 
If satisfied with model performance, the model will be deployed in web application that process the satellite images and give the relevant output.

To provide with more statistical data we have also built an analysis program which takes the images of the same coordinate of different years and it marks the deforested land with red contour. 

This is achieved through open cv library used for image processing. 

This will also integrated be integrated in web application to show the graph and deforestation done over the years.


INNOVATION


We've developed the inaugural model for categorizing various forms of deforestation. 


Our model extracts pertinent information from satellite imagery that may elude human observation in identifying the specific type of deforestation. 


Leveraging machine learning principles, our system continually refines its understanding, ensuring precision in data provision.


OUR MISSION


Integrating this model  with government and deforestation websites to help them with early detection of deforestation with it’s types.

Deploy this model for detecting land use and guide  towards sustainable urban planning practices.

Use this model to detect illegal logging or land clearing taking place. This will help in improving effectiveness of law enforcement actions.

By tracking deforestation trends over time, the model can help identify areas with high deforestation rates and prioritize conservation efforts.



FUTURE BUILDING


We will partner with organizations like government agencies, NGOs, or satellite data providers to access a wider range of high-quality, labeled satellite imagery.
This will improve the model's generalizability and ability to handle deforestation patterns in diverse regions.

Expand beyond just detection and offer deforestation data analytics. Provide insights on deforestation trends, risk areas, and potential environmental impacts.

This could provide basic, advanced, or enterprise-level features with varying pricing structures



						THANKYOU


REGARDS: TEAM CODE MONKS









