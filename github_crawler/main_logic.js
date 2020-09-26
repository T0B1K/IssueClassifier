let $ = (x) => document.getElementById(x);  //für die, die unbedingt JQuery wollen
let $$ = (x) => $(x).value;

//sendRequest("docker-library", "php", "3")
var dataToBeStored = new Set(),
    labelList = $("labelList"),
    labelMapLength = 0,
    labelMap = {},
    requestPage = 1,
    displayPage = 0,
    maxRequest = 6,
    username = "",
    reponame = "";

/**
* This method is setting the initial values
*/
function initValues() {
    dataToBeStored = new Set();
    labelMapLength = 0;
    labelMap = {};
    requestPage = 1;
    displayPage = 0;
}

/**
* This method is starting the whole procedure and is activated, if the user clicks on the "start" button
*/
function clickedOnStart() {
    initValues();
    maxRequest = parseInt($$("txfMaxRequests")) || 6;
    if ($$("inputUserNameAndRepo") !== "") {
        let userRepoNameArray = $$("inputUserNameAndRepo").split("/").map(x => x.trim());
        if (userRepoNameArray.length == 2) {
            username = userRepoNameArray[1].trim();
            reponame = userRepoNameArray[0].trim();
            console.log(username, reponame)
        }
    } else {
        username = $$("inputUserName").trim();
        reponame = $$("inputRepoName").trim();
    }
    sendRequest(username, reponame, getAllLabels())

}
/**
* This mehtod is used to go one requestPage forward or backwards
*/
function nextIssue(nextPage = false) {
    gotoPage(displayPage + (nextPage ? 1 : -1))
}
/**
* This mehtod is used to display the downloaded issue-page
*/
function gotoPage(pageNr) {
    if (dataToBeStored.size <= 0) return
    var data = [...dataToBeStored];
    displayPage = pageNr % data.length;
    if (displayPage < 0)
        displayPage = data.length - displayPage - 2;

    $("txaText").value = data[displayPage].text;
    $("lblOut").value = "labels: " + data[displayPage].labels.join(",");
    $("txfPage").innerText = `  ${displayPage + 1} / ${data.length}  `;
}

/**
* This method is used to create a new html element in the issue mapping list
*/
function createListElement() {
    let li = document.createElement("li"),
        inp1 = document.createElement("input"),
        inp2 = document.createElement("input"),
        div = document.createElement("div");

    inp1.id = `isLbl${++labelMapLength}`;
    inp2.id = `toLbl${labelMapLength}`;
    inp1.placeholder = " istLabel";
    inp2.placeholder = " sollLabel";

    div.classList.add("autocomplete");
    div.appendChild(inp2);
    li.append(inp1);
    li.append(div);

    labelList.appendChild(li);
    labelList.appendChild($("addLabel"));

    autocomplete($(`toLbl${labelMapLength}`), autocompletionTerms);
}

function getAllLabels() {
    for (let i = 0; i <= labelMapLength; i++) {
        let tmp = document.getElementById(`isLbl${i}`).value
        if (!(tmp in labelMap) && (tmp != ""))
            labelMap[tmp] = document.getElementById(`toLbl${i}`).value;
    }
    return Object.keys(labelMap).join(",");
}

function reqListener() {        //falls was schief geht zum debuggen
    let apiResponse = JSON.parse(this.responseText);
    queryResult(apiResponse)
    if (apiResponse.length >= 100 && requestPage <= maxRequest) {
        $("txfCurrentPage").innerText = `#requests:${requestPage} von ${maxRequest}\ngesammelte Issues:${dataToBeStored.size}`;
        ++requestPage
        sendRequest(username, reponame, getAllLabels())
    } else {
        let name = `${username}_${reponame}`;
        if ($("toLbl0").value != "") {
            name += "_" + $("toLbl0").value;
        }
        download(JSON.stringify([...dataToBeStored]), `${name}.json`, "text/plain");
        gotoPage(0);
    }
}

/*
* This method sends a request to github, to the specific repo
*/
function sendRequest(user, repo, lbl = "") {
    let oReq = new XMLHttpRequest(),
        lblQuerry = (lbl != "") ? `&&labels=${lbl}` : "";
    oReq.addEventListener("load", reqListener);
    let requestText = `https://api.github.com/repos/${user}/${repo}/issues?state=all${lblQuerry}&page=${requestPage}&per_page=100`;
    console.log(`requestText: ${requestText}`);
    oReq.open("GET", requestText);
    oReq.send();
}

/*
* This method is being used to put the received data in a readable format 
*/
function queryResult(jsonData) {
    if (jsonData.length == undefined) return
    for (obj of jsonData) {
        if (obj.pull_request || obj.labels.length == 0) {
            console.log("pull");
            continue;
        };
        let { body, labels, ...rest } = obj
        labelNames = labels.map((x) => (Object.keys(labelMap).length > 0) ? labelMap[x.name] : x.name)
        let newJson = { "labels": labelNames, "text": body }
        //console.log(newJson)
        dataToBeStored.add(newJson)
    }
}

/**
This method is being used to download the received data
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