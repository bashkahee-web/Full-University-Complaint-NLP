import express from "express";
import { createUsers, getUsers } from "../controllers/SignUp.js";
import { LoginUser } from "../controllers/Login.js";
import { CompliantsUser, deleteStudentdata, getComplaintsByCategory, getStudentComplaints, updateStatus } from "../controllers/CompliantControl.js";
import { addResponse, deleteInboxdata, getResponsesByComplaint } from "../controllers/ResponseController.js";

const route = express.Router();

route.post("/signup", createUsers);
route.get("/getusers", getUsers);
route.post("/loginuser", LoginUser);
route.post("/complaints", CompliantsUser);
route.get("/complaints/:category", getComplaintsByCategory);
route.put("/status/:id", updateStatus);
route.get('/studentData/:stdId', getStudentComplaints);

route.delete('/delComplaint/:id', deleteStudentdata);

route.post('/responses', addResponse);
route.get('/responses/:UsertId', getResponsesByComplaint);
route.delete('/delResponses/:id', deleteInboxdata);

export default route;
