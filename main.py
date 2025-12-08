from itertools import chain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from urllib3 import response
from langchain_ollama import ChatOllama

#import os
load_dotenv() 

def main():
    print("Hello from langchain-course!")
    #print(os.environ.get("OPENAI_API_KEY"))

information = """ The Doctors Company claims to be the largest physician-owned medical malpractice 
insurer in the U.S., with 97,000 insureds nationwide, $7.8 billion in assets, $3.1 billion in member surplus, and a financial strength rating of A from A. M. Best and Fitch Ratings in fiscal year 2024

Founded in 1976, The Doctors Company operates as a member-owned interinsurance exchange, with a number of wholly owned subsidiaries, and is led by a 15-member Board of Governors.[2]

In 2008, The Doctors Company merged with Los Angeles-based SCPIE Holdings, Inc.[3]

In October 2010, the company completed a $386 million purchase of American Physicians Capital, Inc.[4]

On October 19, 2011, The Doctors Company acquired FPIC Insurance Group, Inc., and its subsidiaries First Professionals Insurance Company, Inc.; Advocate, MD Insurance of the Southwest, Inc.; Anesthesiologists Professional Assurance Company; and Intermed Insurance Company.[5]

On June 17, 2014, The Doctors Company acquired Medical Advantage Group, a provider of healthcare, electronic health record (EHR), and telehealth consulting services.[6] The Doctors Company sold Medical Advantage Group to Aledade on May 1, 2024. [7]

On August 8, 2016, The Doctors Company created an excess and surplus lines subsidiary, TDC Specialty Underwriters, to facilitate expansion of its product and service offerings to a broader and changing array of healthcare industry insurance buyers.[8]

On July 31, 2019, The Doctors Company completed the purchase of Hospitals Insurance Company and FOJP Service Corporation. The transaction created Healthcare Risk Advisors, a new service unit that provides third-party insurance and risk management advisory services.[9]

Collectively, The Doctors Company, TDC Specialty Underwriters, and Healthcare Risk Advisors form the TDC Group of companies (TDC Group).[10]

The Doctors Company has been certified as a Great Place to Work(R) for three years.[11]"""


summary_template = """ given the infromation {information} please summarize it in a single sentence :
1. A short summary of the information
2.two interesting facts about the information
"""


summary_prompt_template = PromptTemplate(
                           input_variables=["information"],
                           template=summary_template
                        )

#llm=ChatOpenAI( temperature=0,model="gpt-5")
llm=ChatOllama( temperature=0,model="gemma3:270m")

chain=summary_prompt_template | llm

response=chain.invoke(input={"information":information})
print(response.content)

if __name__ == "__main__":
    main()
