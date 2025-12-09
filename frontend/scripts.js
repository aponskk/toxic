const textInput = document.getElementById('textInput')
const checkButton = document.getElementById('checkButton')
const resultDiv = document.getElementById('result')

checkButton.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (!text) {
        alert("Введите текст!");
        return;
    }

    resultDiv.innerText = "Загрузка...";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        if (!response.ok) throw new Error("Ошибка сервера");

        const data = await response.json();
        const label = data.prediction.label;
        const score = (data.prediction.score * 100).toFixed(2);

        let message = "";
        if (label === "toxic") {
            message = `Этот текст содержит <span class="toxic-word">негативный</span> характер с вероятностью ${score}%`;
        } else if (label === "neutral") {
            message = `Этот текст имеет нейтральный характер с вероятностью ${score}%`;
        }

        resultDiv.innerHTML = message;

    } catch (err) {
        resultDiv.innerText = "Ошибка: " + err.message;
    }
});
