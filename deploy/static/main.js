//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

function changeDataset() {
    var nSel = document.getElementById('datasetSel');
    var index = nSel.selectedIndex;
    var value = nSel.options[index].value;
    // clear preview
    clearImage()
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var rec_imageDisplay = document.getElementById("rec-image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");
var recResult = document.getElementById("rec-result");

//========================================================================
// Main button events
//========================================================================

function submitImage_CMC() {
  // action for the submit button
  console.log("submit");

  if ((!imageDisplay.src || !imageDisplay.src.startsWith("data")))
    window.alert("Please select an image before submit.");
    return;
  }

  //loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");
  rec_imageDisplay.classList.add('loading')

  // call the predict function of the backend
  predictImage_CMC(imageDisplay.src);
}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  imageDisplay.src = "";
  rec_imageDisplay.src = ''
  predResult.innerHTML = "";

  hide(imagePreview);
  hide(imageDisplay);
  hide(rec_imageDisplay)
  hide(loader);
  hide(predResult);
  hide(recResult)
  show(uploadCaption);

  imageDisplay.classList.remove("loading");
  rec_imageDisplay.classListl.remove("loading")
}

function previewFile(file) {
  // show the preview of the image
  console.log(file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    // reset
    predResult.innerHTML = "";
    imageDisplay.classList.remove("loading");

    recResult.innerHTML = ""
    rec_imageDisplay.classList.remove("loading")
    
    var nSel = document.getElementById('datasetSel');
    var index = nSel.selectedIndex;
    var value = nSel.options[index].value;
    displayImage(reader.result, "image-display");
    displayImage(reader.result, 'rec-image-display')    
  };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage_CMC(image) {
  var nSel = document.getElementById('datasetSel');
  var index = nSel.selectedIndex;
  var value = nSel.options[index].value;

  fetch("/cross_modal_compression_mini/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({'img':image, 'dataset':value})
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  // display the result
  // imageDisplay.classList.remove("loading");
  hide(loader);
  predResult.innerHTML = data.result;
  show(predResult);
  rec_imageDisplay.src = data.rec 
  show(recResult);
  // show(rec_imageDisplay);
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}