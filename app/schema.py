from enum import Enum
from typing import Annotated, List
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator

class ContractOpts(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"

class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"

class YesNo(str, Enum):
    yes = "Yes"
    no = "No"

class YesNo_NoInternet(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"

class YesNo_NoPhone(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"

class GenderOpts(str, Enum):
    male = "Male"
    female = "Female"

class InternetService(str, Enum):
    fiber = "Fiber optic"
    dsl = "DSL"
    no = "No"

class CustomerFeatures(BaseModel):
    Tenure_Months: Annotated[int, Field(..., ge=0, description="Customer tenure in Months")]
    Monthly_Charges: Annotated[float, Field(..., ge=0, description="Monthly charges")]
    CLTV: Annotated[int, Field(..., ge=0, description="Customer lifetime value (as an integer)")]

    Zip_Code: str = Field(
        ...,
        description="5-digit or ZIP+4 postal code",
        pattern=r"^\d{5}(-\d{4})?$"
    )

    Internet_Service: InternetService = Field(..., description="Type of internet service: [Fiber optic|DSL|No]")
    Gender: GenderOpts = Field(..., description="Gender of the customer [Male|Female]")
    Senior_Citizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    Phone_Service: YesNo
    Paperless_Billing: YesNo
    Contract: ContractOpts
    Payment_Method: PaymentMethod
    Multiple_Lines: YesNo_NoPhone
    Online_Security: YesNo_NoInternet
    Online_Backup: YesNo_NoInternet
    Device_Protection: YesNo_NoInternet
    Tech_Support: YesNo_NoInternet
    Streaming_TV: YesNo_NoInternet
    Streaming_Movies: YesNo_NoInternet

    model_config = {
        "populate_by_name": True,
        "use_enum_values": True,
    }

    @model_validator(mode='after')
    def validate_phone_fields(self) -> Self:
        phone_service_dep_fields = [self.Multiple_Lines]
        internet_service_dep_fields = [self.Online_Security,
                                    self.Online_Backup,
                                    self.Device_Protection,
                                    self.Tech_Support,
                                    self.Streaming_TV,
                                    self.Streaming_Movies]

        for f in phone_service_dep_fields:
            if self.Phone_Service == YesNo.no and  f != YesNo_NoPhone.no_phone:
                raise ValueError('Invalid configuration for phone service')
            if self.Phone_Service == YesNo.yes and f == YesNo_NoPhone.no_phone:
                raise ValueError('Invalid configuration for phone service')
        
        for f in internet_service_dep_fields:
            if self.Internet_Service == YesNo.no and  f != YesNo_NoInternet.no_internet:
                raise ValueError('Invalid configuration for phone service')
            if self.Internet_Service == YesNo.yes and  f == YesNo_NoInternet.no_internet:
                raise ValueError('Invalid configuration for phone service')
        return self
            
class PredictOneResponse(BaseModel):
    probability: float

class PredictBatchResponse(BaseModel):
    probabilities: List[float]