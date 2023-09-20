from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from Fucntions import data_ingestion, train_test_split, model_building, model_training, prediction



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['hamza91ghani@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'Image Classifier',
    default_args=default_args,
    start_date=datetime(2023, 9, 19),
    description='A DAG to classify images using DL',
    schedule=timedelta(days=1),
)

def run_data_ingestion():
    data_ingest = data_ingestion('data')
    data = data_ingest.data_ingest()
    return data

def run_tts(**context):
    ti = context['task_instance']
    data = ti.xcom_pull(task_ids='data_ingestion_task')   
    tts=train_test_split(data) 
    train_data, val_data, test_data = tts.tts()
    return train_data,val_data,test_data

def run_model_building():
    model_builder = model_building()
    model = model_builder.model_compile()
    return model

def run_model_training(**context):
    ti = context['task_instance']
    train_data,val_data,test_data = ti.xcom_pull(task_ids='train_test_split')
    model=ti.xcom_pull(task_ids='model_building')
    trainer = model_training(model, train_data, val_data)
    trainer.trained_model()

def run_predictions(**context):
    ti=context['task_instance']
    model = ti.xcom_pull(task_ids='model_building')
    predictor = prediction(model, 'sad.jpg')
    predictor.predict()



T1 = PythonOperator(
    task_id='data_ingestion_task',
    python_callable=run_data_ingestion,
    dag=dag
)

T2 = PythonOperator(
    task_id='train_test_split',
    python_callable=run_tts,
    dag=dag
)

T3 = PythonOperator(
    task_id='model_building',
    python_callable=run_model_building,
    dag=dag
)

T4 = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag
)

T5 = PythonOperator(
    task_id='predictions',
    python_callable=run_predictions,
    dag=dag
)


T1 >> T2 >> T3 >> T4 >> T5
