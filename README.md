# ProxyFloxy: A simple use case of Federated Learning on top of data decoupling with ProxyStore

This repo is a simple app that modifies the `FLoX-v0.1.0` code base to incorporate ProxyStore as a data transfer
protocol.

Below is a high-level visual of how this approach works. In traditional FL, where the aggregator/controller and the
training endpoints directly share raw model parameters with one another. This application instead inserts ProxyStore as
a data decoupling abstraction. This handles the finer details regarding data transfer and ensures that model parameters
have arrived when they are needed for aggregation.

```mermaid
flowchart LR
serv[Controller]
endp1[Training Endpoint 1]
endp2[Training Endpoint 2]
endpn[Training Endpoint n]
data1[(Training Data)]
data2[(Training Data)]
datan[(Training Data)]
prox[[ProxyStore]]

serv-->|params|prox
prox-.->|proxies|serv

prox-->|proxy|endp1
prox-->|proxy|endp2
prox-->|proxy|endpn
endp1-.->|params|prox
endp2-.->|params|prox
endpn-.->|params|prox


endp1===|train/test|data1
endp2===|train/test|data2
endpn===|train/test|datan
```

***

### TODO:

- [ ] Look into the using
  the [EndpointConnector](https://docs.proxystore.dev/main/guides/endpoints/#endpointconnector) for orchestrating
  proxy transfer across all endpoints. Greg suggests to use the `EndpointConnector`.
- [ ] Polish the current code to be more somewhat legible.
- [ ] Implement a more complex model (loosely based on MobileNet but for CIFAR-10). We want a more complex model to
  make the data transfer improvements maybe a bit more interesting.
- [ ] Move from `RedisStore` to `EndpointStore`.

### NOTES:

- Batch size on the raspberry pis (they're v3) needs to be very small (i.e., `1`).

### ProxyStore Notes:

- Proxystore-Endpoint notes: https://docs.proxystore.dev/main/guides/endpoints/
- Greg sent:

> This is the relay server address that is used with the ProxyStore-Endpoint CLI:
> - wss://relay.proxystore.dev
>
> FuncX client:
> https://github.com/proxystore/benchmarks/blob/main/psbench/benchmarks/funcx_tasks/main.py
> FuncX task:
> https://github.com/proxystore/benchmarks/blob/a157bfad270a9cf7d9fe2cc38f6f9d434d8de48d/psbench/tasks/pong.py#L74
>
> `proxystore-endpoint configure my-endpoint --relay-server wss://relay.proxystore.dev`
>
> (Every note can have the same name, they'll get their own unique uuids.)

To use one of the RPI endpoints, you need to start the `proxystore-endpoint` so it that the `EndpointConnector` for data transfer can work. To do this, run the following in the terminal of the logged in node:
```
proxystore-endpoint start flox
```