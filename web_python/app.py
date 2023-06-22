# uvicorn app:app --host 0.0.0.0 --port 8081

from fastapi import FastAPI, Form
import csv

app = FastAPI()

def save_to_csv(data):
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

@app.get("/")
def home():
    return """
    <html>
    <head>
        <title>Ввод данных</title>
    </head>
    <body>
        <h1>Введите данные</h1>
        <form action="/process_input" method="post">
            <label for="name">Имя:</label>
            <input type="text" id="name" name="name" required><br><br>

            <label for="age">Возраст:</label>
            <input type="number" id="age" name="age" required><br><br>

            <input type="submit" value="Отправить">
        </form>
    </body>
    </html>
    """

@app.post("/process_input")
def process_input(name: str = Form(...), age: int = Form(...)):
    data = [name, age]
    save_to_csv(data)
    return {"message": "Данные успешно сохранены в CSV"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
