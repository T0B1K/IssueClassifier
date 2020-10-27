let uploadedData = null; //Data from the uploaded issue json.

/**
 * Allows the user to load a file into the analysis tool.
 */
function uploadedFile() {
    let file = elemById("fileUploader").files[0],
        fileReader = new FileReader();

    fileReader.readAsText(file);
    fileReader.onload = function () {
        uploadedData = JSON.parse(fileReader.result)
        let labelsSet = new Set();
        labelsSet.add(uploadedData.map(x => x.labels).flat(1))
        labelsSet.delete(null);
        if (labelsSet.size == 1) {
            let set = new Set();
            uploadedData.map(x => x.classified_as).forEach(x => set.add(x))
            labelsSet = set;
        }
        displayIssue = 0;
        let texts = uploadedData.map(x => { return { "text": x.text, "labels": ["klassifiziert als " + x.classified_as] } })
        dataToBeStored = new Set();
        texts.forEach(x => dataToBeStored.add(x))
        elemById("txfIssueCount").innerHTML = `Labels: ${[...labelsSet]}<br>`;
    };
}