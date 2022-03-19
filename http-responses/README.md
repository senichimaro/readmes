# HTTP 

## Making a Request
Example Request:
```
GET / HTTP/1.1
Host: tryhackme.com
User-Agent: Mozilla/5.0 Firefox/87.0
Referer: https://tryhackme.com/
To breakdown each line of this request:
```
* Line 1: This request is sending the GET method ( more on this in the HTTP Methods task ), request the home page with / and telling the web server we are using HTTP protocol version 1.1.

* Line 2: We tell the web server we want the website tryhackme.com

* Line 3: We tell the web server we are using the Firefox version 87 Browser

* Line 4: We are telling the web server that the web page that referred us to this one is `https://tryhackme.com`

* Line 5: HTTP requests always end with a blank line to inform the web server that the request has finished.


## Example Response:

```
HTTP/1.1 200 OK
Server: nginx/1.15.8
Date: Fri, 09 Apr 2021 13:34:03 GMT
Content-Type: text/html
Content-Length: 98

<html>
<head>
    <title>TryHackMe</title>
</head>
<body>
    Welcome To TryHackMe.com
</body>
</html>
```

To breakdown each line of the response:

* Line 1: HTTP 1.1 is the version of the HTTP protocol the server is using and then followed by the HTTP Status Code in this case "200 Ok" which tells us the request has completed successfully.

* Line 2: This tells us the web server software and version number.

* Line 3: The current date, time and timezone of the web server.

* Line 4: The Content-Type header tells the client what sort of information is going to be sent, such as HTML, images, videos, pdf, XML.

* Line 5: Content-Length tells the client how long the response is, this way we can confirm no data is missing.

* Line 6: HTTP response contains a blank line to confirm the end of the HTTP response.

* Lines 7-14: The information that has been requested, in this instance the homepage.



# Headers
Headers are additional bits of data you can send to the web server when making requests.

Although no headers are strictly required when making a HTTP request, you’ll find it difficult to view a website properly.

## Common Request Headers

These are headers that are sent from the client (usually your browser) to the server.

### Host:
Some web servers host multiple websites so by providing the host headers you can tell it which one you require, otherwise you'll just receive the default website for the server.

### User-Agent: 
This is your browser software and version number, telling the web server your browser software helps it format the website properly for your browser and also some elements of HTML, JavaScript and CSS are only available in certain browsers.

### Content-Length: 
When sending data to a web server such as in a form, the content length tells the web server how much data to expect in the web request. This way the server can ensure it isn't missing any data.

### Accept-Encoding: 
Tells the web server what types of compression methods the browser supports so the data can be made smaller for transmitting over the internet.


### Cookie: 
Data sent to the server to help remember your information (see cookies task for more information).



## Common Response Headers
These are the headers that are returned to the client from the server after a request.

### Set-Cookie: 
Information to store which gets sent back to the web server on each request (see cookies task for more information).

### Cache-Control: 
How long to store the content of the response in the browser's cache before it requests it again.

### Content-Type: 
This tells the client what type of data is being returned, i.e., HTML, CSS, JavaScript, Images, PDF, Video, etc. Using the content-type header the browser then knows how to process the data.

### Content-Encoding: 
What method has been used to compress the data to make it smaller when sending it over the internet.




# HTTP Responses

### 100-199 - Information Response
These are sent to tell the client the first part of their request has been accepted and they should continue sending the rest of their request. These codes are no longer very common.

### 200-299 - Success
This range of status codes is used to tell the client their request was successful.

### 300-399 - Redirection
These are used to redirect the client's request to another resource. This can be either to a different webpage or a different website altogether.

### 400-499 - Client Errors
Used to inform the client that there was an error with their request.

### 500-599 - Server Errors
This is reserved for errors happening on the server-side and usually indicate quite a major problem with the server handling the request.

![http resume](http-resume.png)


# Common HTTP Status Codes:
There are a lot of different HTTP status codes and that's not including the fact that applications can even define their own, we'll go over the most common HTTP responses you are likely to come across:

## 200 - OK
The request was completed successfully.

## 201 - Created
A resource has been created (for example a new user or new blog post).

## 301 - Permanent Redirect
This redirects the client's browser to a new webpage or tells search engines that the page has moved somewhere else and to look there instead.

## 302 - Temporary Redirect
Similar to the above permanent redirect, but as the name suggests, this is only a temporary change and it may change again in the near future.

## 400 - Bad Request
This tells the browser that something was either wrong or missing in their request. This could sometimes be used if the web server resource that is being requested expected a certain parameter that the client didn't send.

## 401 - Not Authorised
You are not currently allowed to view this resource until you have authorised with the web application, most commonly with a username and password.

## 403 - Forbidden
You do not have permission to view this resource whether you are logged in or not.

## 405 - Method Not Allowed
The resource does not allow this method request, for example, you send a GET request to the resource /create-account when it was expecting a POST request instead.

## 404 - Page Not Found
The page/resource you requested does not exist.

## 500 - Internal Service Error
The server has encountered some kind of error with your request that it doesn't know how to handle properly.

## 503 - Service Unavailable	
This server cannot handle your request as it's either overloaded or down for maintenance.

![common responses](common-responses.png)