-- create a triggger to reset valid_email only if email is updated
DROP TRIGGER IF EXISTS reset_valid_email;

DELIMITER //
CREATE TRIGGER reset_valid_email BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NEW.email != OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END;
//
DELIMITER ;
 
--  This time, we have to create a trigger that will reset the  valid_email  column to  0  if the  email  column is updated. 
--  The trigger is called  reset_valid_email  and is created on the  users  table. It is a  BEFORE UPDATE  trigger, which means that it will be executed before the update is performed. 
--  The trigger checks if the  email  column has been updated. If it has, the  valid_email  column is set to  0 . 
--  The  DELIMITER  command is used to change the delimiter from  ;  to  //  and back to  ;  at the end. This is necessary because the trigger definition contains semicolons. 
--  Create a trigger that will reset the  valid_email  column to  0  if the  email  column is updated. 