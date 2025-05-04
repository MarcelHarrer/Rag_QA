const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const uploadButton = document.getElementById("upload-button");

let files = [];

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragging");
    addFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
});

function addFiles(newFiles) {
    for (const file of newFiles) {
        if (!files.some((f) => f.name === file.name)) {
            files.push(file);
            const listItem = document.createElement("li");
            listItem.innerHTML = `<span>${file.name}</span> <button onclick="removeFile('${file.name}')">Remove</button>`;
            fileList.appendChild(listItem);
        }
    }
}

function removeFile(fileName) {
    files = files.filter((file) => file.name !== fileName);
    const listItems = Array.from(fileList.children);
    for (const item of listItems) {
        if (item.querySelector("span").textContent === fileName) {
            fileList.removeChild(item);
            break;
        }
    }
}

uploadButton.addEventListener("click", () => {
    if (files.length === 0) {
        alert("Please select or drag and drop files to upload.");
        return;
    }

    const formData = new FormData();
    for (const file of files) {
        formData.append("files", file);
    }

    fetch("/upload-pdf/", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.text())
        .then((html) => {
            document.body.innerHTML = html;
        })
        .catch((error) => console.error("Error:", error));
});
