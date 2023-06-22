from datetime import datetime
from airflow import DAG
# from airflow.decorators import task
from airflow.operators.bash import BashOperator

# A DAG represents a workflow, a collection of tasks
dag =  DAG(dag_id="demo", start_date=datetime(2022, 1, 1))

# Tasks are represented as operators
operation = BashOperator(task_id="hello", bash_command="HELLO WORLD", dag=dag)

operation