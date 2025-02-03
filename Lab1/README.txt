Here's the link to my Docker image: https://hub.docker.com/r/candykane2/image-classification 

Instructions for Running the Container Locally

To start, to run the container locally, you first need to pull my container from Docker Hub at the following link: https://hub.docker.com/r/candykane2/image-classification.
Then, to run the image, use the following command: docker run -p 5000:80 candykane2/image-classification:1.0 (or docker run -p 5000:80 candykane2/image-classification)
(Unsure on if the name will be the same. If not the same it's same but replace candykane2... with the name or id of the image)
After executing this command, your container should be running locally. 

There are two ways to get predictions for sample inputs. In both cases, the input is a URL string pointing to an image.

Method 1: Using the UI

You should be able to access the UI at http://localhost:5000/, where you will see the same interface as shown in apiEndpointRunning.png. To use it:
    1. Enter the image URL in the text field.
    2. Click the "Predictions" button.
    3. The output will be displayed in JSON format on the UI, similar to predictionUI.png.

Method 2: Using the /predict API Endpoint (cURL Command)

You can also get predictions by making a POST request to the /predict endpoint as shown in predictionCMD.png.
    1. For example, using curl: curl -X POST -H "Content-Type: text/plain" -d "IMAGE_URL" http://localhost:5000/predict
       Replace IMAGE_URL with the actual image URL you want to classify.
    2. After running this command, the response will be printed in the terminal (CMD).