<!DOCTYPE html>
<html lang="fr">

<head>
  <meta charset="UTF-8">
  <title>Chatbot</title>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <link rel="stylesheet" href="{{ url_for('static', filename='styles/style.css') }}">
  <link rel="stylesheet" href="../static/styles/style.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
</head>

<body>
  <!-- partial:index.partial.html -->
  <section class="msger">
    <header class="msger-header">
      <div class="msger-header-title">
         <h2><i class="fas fa-robot"></i></h2>
      </div>
    </header>

    <script src='https://use.fontawesome.com/releases/v5.0.13/js/all.js'></script>
  <script>
    <main class="msger-chat">
      <div class="msg left-msg">
        <div class="msg-img" style="background-image: url(https://i.pinimg.com/736x/7d/9b/1d/7d9b1d662b28cd365b33a01a3d0288e1--robot-logo-design-logo-robot.jpg)"></div>

        <div class="msg-bubble">
          <div class="msg-info">
            <div class="msg-info-name">Frenchbot</div>
            <div class="msg-info-time">`${formatDate(new Date())}`</div>
          </div>

          <div class="msg-text">
            Bienvenue dans notre platfome d'apprentissage de la langue francaise.
            <br>
            Nous avons une large choix pour vous, des cours, des ressouces, des conseils..
            <br>
            N'hesitez pas a poser les questions, je suis la pour vous aider ! 😄
          </div>
        </div>
      </div>

    </main>

    <form class="msger-inputarea">
      <input type="text" class="msger-input" id="textInput" placeholder="Enter your message...">
      <button type="submit" class="msger-send-btn">Send</button>
    </form>
  </section>
  <!-- partial -->
  

    const msgerForm = get(".msger-inputarea");
    const msgerInput = get(".msger-input");
    const msgerChat = get(".msger-chat");


    const BOT_IMG = "https://i.pinimg.com/736x/7d/9b/1d/7d9b1d662b28cd365b33a01a3d0288e1--robot-logo-design-logo-robot.jpg";
    const PERSON_IMG = "https://th.bing.com/th/id/R.7f0a40dc5c8a5d5cde4aec91aa3a62df?rik=xvmDdnzDfjotog&pid=ImgRaw&r=0";
    const BOT_NAME = "    FrenchBot";
    const PERSON_NAME = "You";

    msgerForm.addEventListener("submit", event => {
      event.preventDefault();

      const msgText = msgerInput.value;
      if (!msgText) return;

      appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
      msgerInput.value = "";
      botResponse(msgText);
    });

    function appendMessage(name, img, side, text) {
      //   Simple solution for small apps
      const msgHTML = `
<div class="msg ${side}-msg">
  <div class="msg-img" style="background-image: url(${img})"></div>

  <div class="msg-bubble">
    <div class="msg-info">
      <div class="msg-info-name">${name}</div>
      <div class="msg-info-time">${formatDate(new Date())}</div>
    </div>

    <div class="msg-text">${text}</div>
  </div>
</div>
`;

      msgerChat.insertAdjacentHTML("beforeend", msgHTML);
      msgerChat.scrollTop += 500;
    }

    function botResponse(rawText) {

      // Bot Response
      $.get("/get", { msg: rawText }).done(function (data) {
        console.log(rawText);
        console.log(data);
        const msgText = data;
        appendMessage(BOT_NAME, BOT_IMG, "left", msgText);

      });

    }


    // Utils
    function get(selector, root = document) {
      return root.querySelector(selector);
    }

    function formatDate(date) {
      const h = "0" + date.getHours();
      const m = "0" + date.getMinutes();

      return `${h.slice(-2)}:${m.slice(-2)}`;
    }



  </script>

</body>

</html>