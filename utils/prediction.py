# OfferSense AI
# 

import numpy as np

# Predict Accept / Reject

def predict_offer(model, data):

    prediction = int(
        model.predict(data)[0]
    )

    probability = model.predict_proba(data)[0]


    # ------------------------------------------
    # Map probabilities using actual model classes
    # ------------------------------------------

    class_probabilities = dict(
        zip(
            model.classes_,
            probability
        )
    )


    # ------------------------------------------
    # Reject Probability
    # ------------------------------------------

    rejection_probability = (
        class_probabilities.get(0, 0.0) * 100
    )


    # ------------------------------------------
    # Accept Probability
    # ------------------------------------------

    acceptance_probability = (
        class_probabilities.get(1, 0.0) * 100
    )


    return (
        prediction,
        acceptance_probability,
        rejection_probability
    )
    
# Convert Prediction to Label

def prediction_label(prediction):

    if prediction == 1:
        return "Accept"

    return "Reject"


# HR Recommendation

def hr_recommendation(
    prediction,
    acceptance_probability
):

    if prediction == 1:

        if acceptance_probability >= 90:

            return """
Candidate is highly likely to accept the offer.

- Proceed with the hiring process immediately.
- Send the offer letter as soon as possible.
- Candidate is an excellent hiring opportunity.
"""

        elif acceptance_probability >= 70:

            return """
Candidate is likely to accept the offer.

- Follow up with the candidate.
- Maintain regular communication.
- HR can confidently proceed.
"""

        else:

            return """
Candidate may accept the offer.

- Consider improving the salary package.
- Discuss additional benefits before releasing the offer.
"""

    else:

        return """
Candidate is unlikely to accept the offer.

- Review salary expectations.
- Consider negotiating compensation.
- Reconnect with the candidate before closing the position.
"""