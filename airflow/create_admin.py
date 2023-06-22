# import airflow
# from airflow import models, settings
# from airflow.contrib.auth.backends.password_auth import PasswordUser

# user = PasswordUser(models.User())
# user.username = 'daniil_admin'
# user.email = 'daniil_325@mail.ru'
# user.password = '12345'
# user.superuser = True
# session = settings.Session()
# session.add(user)
# session.commit()
# session.close()
# exit()