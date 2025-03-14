$(document).ready(function () {
  $("#translateBtn").click(function () {
    const text = $("#sourceText").val();
    const sourceLang = $("#sourceLang").val();
    const targetLang = $("#targetLang").val();

    if (text.trim() === "") {
      alert("Please enter text to translate.");
      return;
    }

    fetch('/translate', {
      method: 'POST',
      body: JSON.stringify({
        sourceLang, targetLang
      })
    }).then(res => res.text())
      .then(res => {
      $('#output').val(res)
    })

  });
});