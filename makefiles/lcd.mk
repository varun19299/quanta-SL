# lcd.%: Generic lcd make command
#lcd.%:
#	${PYTHON} quanta_SL/lcd/$*.py

## lcd.calibrate.% : Generic lcd calibrate command
lcd.calibrate.%:
	${PYTHON} quanta_SL/lcd/calibrate/$*.py $(HYDRA_FLAGS)


## lcd.acquire.%: LCD acquire commands
lcd.calibrate.%:
	${PYTHON} quanta_SL/lcd/acquire/$*.py $(HYDRA_FLAGS)

## lcd_calibrate: Run all calibration steps
LCD_CALIBRATE_DEPS := lcd.calibrate.decode_correspondences
LCD_CALIBRATE_DEPS += lcd.calibrate.get_intrinsic_extrinsic
LCD_CALIBRATE_DEPS += lcd.calibrate.reconstruct
lcd_calibrate: $(LCD_CALIBRATE_DEPS)

