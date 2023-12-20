# RD-NL-Takehome-Fahimeh-Ebrahimi
I used Google Colab for this assessment task.

model.pt:
Due to Github file size limit, I had to upload the trained model on My Google Drive, here is the link:
https://drive.google.com/file/d/1MTs58qYHtnA7FCGueNjTbAciPWwjwSDr/view?usp=drive_link

main.py:
  Predict the labels of an unseen dataset (use --help for more options).
	
	python main.py \
	--model_pt "/model/model.pt" \
	--test_data_path "/data/rd-gen-text-toy-dataset - test.csv"
	

train.py:
	Train and save the model (use --help for more options)
	
	python train.py \
	--traindatapath "/data/rd-gen-text-toy-dataset - train.csv" \
	--validationdatapath "/data/rd-gen-text-toy-dataset - validation.csv" \
	--epochs 5 \
	--learning_rate 1e-6\
	--modelpath "/model/" 
	
