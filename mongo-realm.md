# Mongo Real

0. Section "BUILD"
1. 3rd Party Services (renamed HTTPS Endpoints)
2. Add a Service : HTTP
3. Service Name
4. Add Incoming Webhook
5. Name (endpoint name)
6. Select HTTP Method
7. url : "webhook URL"


## POST
Special for Mongo Real we have to parse `payload.body.text()` from EJSON (like JSON with extra data) to get the data.
```
const body = EJSON.parse(payload.body.text())
# now we can use the variable 'body' as an regular object.

const myCollection = context.services.get("mongodb-atlas").db("crud-panel").collection("dummy-user");
# now we are connected to the database and the target collection

const userData = {
    name: body.name,
    userID: body.userID,
    text: body.text
}

return await myCollection.insertOne(userData);
```