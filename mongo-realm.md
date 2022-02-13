# Mongo Realm
Simple example for react with `realm-web` library based in anonimous authentication. It's works over only one realm user and always use the same credetials to any operation.
1. set id
2. load credentials for anonimous interactions
3. log in
4. connect to mongodb
5. locate database & table
6. perform operation (create, read, update or delete)
```
const buttonCall = async (event) => {
        event.preventDefault();
        const data = {
            media_type: event.target.value,
            movieID: event.target.id,
            userID: "9"
        }
        // 1.
        const app = await new Realm.App({ id: "crud-panel-backend-ytuar" });
        // 2.
        const credentials = await Realm.Credentials.anonymous();
        // 3.
        const user = await app.logIn(credentials);
        // 4.
        const mongodb = await app.currentUser.mongoClient('mongodb-atlas')
        // 5.
        const tasksCollection = await mongodb.db('crud-panel').collection('dummy-user')
        // 6.
        const insertResult = await tasksCollection.insertOne(data)
    }
```