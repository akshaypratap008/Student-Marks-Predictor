from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    obj = DataIngestion(
        host='127.0.0.1', 
        user='root', 
        password='', 
        database='mlproject1'
    )

    train_data, test_data= obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    result = model_trainer.initiate_model_training(train_arr, test_arr)

    return result