$(document).ready(function () {
  $("#translateBtn").click(function () {
    var text = $("#sourceText").val();
    var sourceLang = $("#sourceLang").val();
    var targetLang = $("#targetLang").val();

    if (text.trim() === "") {
      alert("Please enter text to translate.");
      return;
    }


  });
});