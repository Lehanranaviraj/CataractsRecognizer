import * as React from "react";
import { Box, Card, CardContent, Typography } from "@mui/material";
import { styled } from "@mui/material/styles";

const PlaceholderImage = styled("div")({
  border: "3px dashed #44CBCB",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  color: "#aaa",
  borderRadius: "10px",
  width: "100%",
  height: "100%",
});

const placeholderStyle = {
  width: "400px",
  height: "200px",
};

export default function MediaCard({ apiResponse }) {
  console.log("medicard", apiResponse);

  // Ensure apiResponse is not null before accessing its properties
  if (!apiResponse) {
    return (
      <Card sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "400px", padding: "16px" }}>
        <CardContent sx={{ textAlign: "center", fontWeight: "bold", letterSpacing: "5px" }}>
          <Typography variant="h4">The Result</Typography>
          <Typography variant="subtitle1" color="text.secondary" style={{ marginBottom: "16px", marginTop: "10px" }}>
            No results available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "400px", padding: "16px" }}>
      <CardContent sx={{ textAlign: "center", fontWeight: "bold", letterSpacing: "5px" }}>
        <Typography variant="h4">The Result</Typography>
        {apiResponse?.predicted_class && (
          <Typography variant="subtitle1" color="text.secondary" style={{ marginBottom: "16px", marginTop: "10px" }}>
            Check your results
          </Typography>
        )}
        {apiResponse?.predicted_class && (
          <>
            <Typography variant="body1">Prediction: {apiResponse.predicted_class}</Typography>
            <Typography variant="body2" color="text.secondary">
              Confidence: {apiResponse.confidence}
            </Typography>
          </>
        )}
      </CardContent>
      <Box sx={{ width: "60%", display: "flex", justifyContent: "center" }}>
        {apiResponse && apiResponse.img ? (
          <img
            src={`data:image/png;base64,${apiResponse.img}`} // Display the base64 image
            alt="Result"
            style={{ maxWidth: "100%", maxHeight: "100%" }}
          />
        ) : (
          <PlaceholderImage style={placeholderStyle}>The Result will be displayed here...</PlaceholderImage>
        )}
      </Box>
    </Card>
  );
}
