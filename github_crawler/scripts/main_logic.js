var dataToBeStored = new Set(),         // Contains the crawled issues.
    labelList = elemById("labelList"),  // Contains the list of all the labels the issues are filtered by. 
    labelMapLength = 0,                 // The amount of specified labels.
    labelMap = {},                      // The set of labels to which the original labels are mapped.
    requestPage = 1,                    // The page number of the requested issue page.
    displayIssue = 0,                   // The number of the currently displayed issue.
    maxRequest = 6,                     // The max. number of requests.
    username = "",                      // The owner of the GitHub Repostory.
    reponame = "";                      // The name of the GitHub Repository.

/**
 * This method is setting the initial values.
 */
function initValues() {
    dataToBeStored = new Set();
    labelMapLength = elemById("labelList").getElementsByTagName("li").length - 1;
    labelMap = {};
    requestPage = 1;
    displayIssue = 0;
}

/**
 * This method is starting the whole procedure and is activated, if the user clicks on the "start" button.
 */
function clickedOnStart() {
    initValues();
    maxRequest = parseInt(elemByIdValue("txfMaxRequests")) || 6;
    username = elemByIdValue("inputUserName").trim();
    reponame = elemByIdValue("inputRepoName").trim(); 
    sendRequest(username, reponame, getAllLabels());
}

/**
 * This mehtod is used to go one requestPage forward or backwards.
 */
function nextIssue(nextPage = false) {
    gotoPage(displayIssue + (nextPage ? 1 : -1))
}

/**
 * This mehtod is used to display the downloaded issue-page.
 */
function gotoPage(pageNr) {
    if (dataToBeStored.size <= 0) return
    var data = [...dataToBeStored];
    displayIssue = pageNr % data.length;
    if (displayIssue < 0)
        displayIssue = data.length - displayIssue - 2;

    elemById("txaText").value = data[displayIssue].text;
    elemById("lblOut").value = data[displayIssue].labels.join(",");
    elemById("txfPage").innerText = `  ${displayIssue + 1} / ${data.length}  `;
}

/**
 * This method is used to create a new html element in the issue mapping list.
 */
function createListElement() {
    /*Creates the DOM elements for the new item:*/
    let li = document.createElement("li"),
        inp1 = document.createElement("input"),
        inp2 = document.createElement("input"),
        div = document.createElement("div");

    /*Increases the counter of the list elemnts and gives the new element an unique name:*/
    labelMapLength ++;
    inp1.id = `isLbl${labelMapLength}`;
    inp2.id = `toLbl${labelMapLength}`;
    inp1.placeholder = "hat Label";
    inp2.placeholder = "Label umbenennen zu...";

    /*Adds the autocomplete function to the new element:*/
    div.classList.add("autocomplete");
    div.appendChild(inp2);
    li.append(inp1);
    li.append(div);

    labelList.appendChild(li);
    labelList.appendChild(elemById("addLabel"));

    autocomplete(elemById(`toLbl${labelMapLength}`), autocompletionTerms);
}

/**
 * This method returns all the labels the user wants to crawl.
 */
function getAllLabels() {
    /*The loop iterates over the list to get all given labels:*/
    for (let i = 0; i <= labelMapLength; i++) {
        let tmp = document.getElementById(`isLbl${i}`).value
        if (!(tmp in labelMap) && (tmp != ""))
            labelMap[tmp] = document.getElementById(`toLbl${i}`).value;
    }
    return Object.keys(labelMap).join(",");
}

/**
 * Processes the request results and prepares further requests if necessary. 
 */
function reqListener() {
    let apiResponse = JSON.parse(this.responseText);
    queryResult(apiResponse)
    /*Sends a request if there are at least hundred issues left and if the maximum number of requests has not yet been reached:*/
    if (apiResponse.length >= 100 && requestPage <= maxRequest) {
        elemById("requestStatus").innerHTML = `Requests:${requestPage} von ${maxRequest}<br>Gesammelte Issues:${dataToBeStored.size}<br><br>`;
        ++requestPage
        sendRequest(username, reponame, getAllLabels())
    } else {
        let name = `${username}_${reponame}`;
        if (elemById("toLbl0").value != "") {
            name += "_" + elemById("toLbl0").value;
        }
        /*Downloads the JSON file with the issues:*/
        download(JSON.stringify([...dataToBeStored]), `${name}.json`, "text/plain");
        gotoPage(0);
    }
}

/**
 * This method sends a request to github, to the specific repo.
 */
function sendRequest(user, repo, lbl = "") {
    let oReq = new XMLHttpRequest(),
    lblQuerry = (lbl != "") ? `&&labels=${lbl}` : "";
    oReq.addEventListener("load", reqListener);
    let requestText = `https://api.github.com/repos/${user}/${repo}/issues?state=all${lblQuerry}&page=${requestPage}&per_page=100`;
    oReq.open("GET", requestText);
    oReq.send();
}

/**
 * This method is being used to put the received data in a readable format. 
 */
function queryResult(jsonData) {
    if (jsonData.length == undefined) return
    for (obj of jsonData) {
        if (obj.pull_request || obj.labels.length == 0) {
            continue;
        };
        let { body, labels, ...rest } = obj
        labelNames = labels.map((x) => (Object.keys(labelMap).length > 0) ? labelMap[x.name] : x.name)
        let newJson = { "labels": labelNames, "text": body }
        dataToBeStored.add(newJson)
    }
}

/**
 * This method is being used to download the received data.
 */
function download(data, filename, type) {
    let file = new Blob([data], { type: type });
    if (window.navigator.msSaveOrOpenBlob)
        window.navigator.msSaveOrOpenBlob(file, filename);
    else {
        let a = document.createElement("a"),
            url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function () {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 0);
    }
}